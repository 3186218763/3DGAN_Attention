import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    """
    用于Canvas2Picture做残差连接
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class PixelShuffleBlock(nn.Module):
    """
    用于Canvas2Picture做上采样
    """

    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


class Canvas2Picture(nn.Module):
    def __init__(self, H, W, input_dim, initial_channels=128, output_channels=3):
        super().__init__()
        self.initial_channels = initial_channels
        self.H = H
        self.W = W

        self.fc = nn.Linear(input_dim, initial_channels * (H // 8) * (W // 8))
        self.res_block1 = ResidualBlock(initial_channels)
        self.res_block2 = ResidualBlock(initial_channels)

        self.pixel_shuffle1 = PixelShuffleBlock(initial_channels, initial_channels // 2, upscale_factor=2)
        self.pixel_shuffle2 = PixelShuffleBlock(initial_channels // 2, initial_channels // 4, upscale_factor=2)
        self.pixel_shuffle3 = PixelShuffleBlock(initial_channels // 4, output_channels, upscale_factor=2)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, input_dim)
        batch_size = x.size(0)

        # 全连接层将 x 变为 (batch_size, initial_channels * (H // 8) * (W // 8))
        x = self.fc(x)

        # 重塑为 (batch_size, initial_channels, H // 8, W // 8)
        x = x.view(batch_size, self.initial_channels, self.H // 8, self.W // 8)

        # 应用残差块
        x = self.res_block1(x)
        x = self.res_block2(x)

        # 应用 PixelShuffle 上采样
        x = self.pixel_shuffle1(x)
        x = self.pixel_shuffle2(x)
        x = self.pixel_shuffle3(x)

        # 控制输出范围(-1. 1)
        x = self.tanh(x)
        return x


class AvgPooling(nn.Module):
    """
    对于图片一般的seq_len太大了，于是需要减小
    """

    def __init__(self, reduction_factor=2):
        super().__init__()
        self.reduction_factor = reduction_factor

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        num_output_tokens = seq_len // self.reduction_factor

        pooled_output = torch.mean(x.view(batch_size, num_output_tokens, self.reduction_factor, embed_dim), dim=2)

        return pooled_output


class DrawCanvas(nn.Module):
    """
    对Canvas进行Self-Attention进行画画,
    """

    def __init__(self, embed_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.ReLU(),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)]
        )

    def forward(self, x):
        canvas = x
        for layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
            attn_output, _ = layer(canvas, canvas, canvas)
            canvas = canvas + attn_output
            canvas = norm(canvas)
            ffn_output = ffn(canvas)
            canvas = canvas + ffn_output
            canvas = norm(canvas)
        return canvas


class PositionalEncoding(nn.Module):
    """
    对 blank_canvas 做位置编码
    """

    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim

        # 计算位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 在 batch 维度上增加维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.to(x.device)  # 将位置编码移动到输入张量的设备
        # 添加位置编码到输入张量
        x = x + pe[:, :x.size(1)]
        return x


class Canvas(nn.Module):
    """
    对空白Canvas和camera做交叉注意力，得到具有位置信息的Canvas
    """

    def __init__(self, embed_dim, num_heads, dropout, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
             for _ in range(num_layers)]
        )

    def forward(self, camera, blank_canvas):
        canvas = blank_canvas
        for layer in self.layers:
            canvas, _ = layer(canvas, camera, camera)
        canvas = blank_canvas + canvas

        return canvas


class Attention_3DGS_Generator(nn.Module):
    def __init__(self, embed_dim, H, W, num_reduction):
        super().__init__()
        self.embed_dim = embed_dim
        self.H = H
        self.W = W
        self.reduction_factor = 2 ** num_reduction

        self.positional_encoding = PositionalEncoding(self.embed_dim, self.H * self.W)
        self.canvas = Canvas(self.embed_dim, num_heads=2, dropout=0.3, num_layers=2)
        self.canvas_pool = nn.Sequential(
            *[AvgPooling() for _ in range(num_reduction)]
        )
        self.draw_canvas = DrawCanvas(self.embed_dim, num_heads=2, dropout=0.3, num_layers=2)
        self.canvas2picture = Canvas2Picture(self.H, self.W,
                                             (self.H * self.W * self.embed_dim) // self.reduction_factor)

    def forward(self, camera, blank_canvas):
        batch_size = camera.size(0)
        camera = camera.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        blank_canvas = blank_canvas.view(batch_size, self.H * self.W, self.embed_dim)  # 展平并确保形状一致
        blank_canvas = self.positional_encoding(blank_canvas)
        canvas = self.canvas(camera, blank_canvas)
        canvas = self.canvas_pool(canvas)
        canvas = self.draw_canvas(canvas)
        canvas = canvas.view(batch_size, -1)
        picture = self.canvas2picture(canvas)

        return picture


class Attention_3DGS_Discriminator(nn.Module):
    def __init__(self, input_channels=3, feature_dim=64):
        super().__init__()

        self.model = nn.Sequential(
            # 输入层 (input_channels, 288, 384)
            nn.Conv2d(input_channels, feature_dim, kernel_size=4, stride=2, padding=1),  # (feature_dim, 144, 192)
            nn.LeakyReLU(0.2, inplace=True),

            # 隐藏层1
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),  # (feature_dim * 2, 72, 96)
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 隐藏层2
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            # (feature_dim * 4, 36, 48)
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 隐藏层3
            nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),
            # (feature_dim * 8, 18, 24)
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 隐藏层4
            nn.Conv2d(feature_dim * 8, feature_dim * 16, kernel_size=4, stride=2, padding=1),
            # (feature_dim * 16, 9, 12)
            nn.BatchNorm2d(feature_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层
            nn.Conv2d(feature_dim * 16, 1, kernel_size=4, stride=1, padding=0),  # (1, 6, 9)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_path = './data/images'
    img_view_path = './data/inputs/images.csv'
    cameras_path = './data/inputs/cameras.csv'
    scale_factor = 0.125
    embed_dim = 32
    W = 3072
    H = 2304
    batch = 4
    blank_canvas = torch.randn((batch, 288, 384, embed_dim), dtype=torch.float32).to(device)
    gen = Attention_3DGS_Generator(embed_dim=embed_dim, H=288, W=384, num_reduction=12).to(device)  # (384, 288)

    cameras = torch.randn(batch, 10).to(device)
    out = gen(cameras, blank_canvas)
    print(out.shape)
