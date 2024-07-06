import torch.nn as nn
import torch
import torch.optim as optim


class ResidualBlock(nn.Module):
    """
    用于Canvas2Picture做残差连接
    """

    def __init__(self, channels):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        out = self.res(x)
        out += x
        return out


class PixelShuffleBlock(nn.Module):
    """
    用于Canvas2Picture做上采样
    """

    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pixel_shuffle(x)
        return x


class Canvas2Picture(nn.Module):
    def __init__(self, H, W, input_dim, initial_channels=128, output_channels=3):
        super().__init__()
        self.initial_channels = initial_channels
        self.H = H
        self.W = W

        self.fc = nn.Linear(input_dim, initial_channels * (H // 8) * (W // 8))
        self.res_block = nn.Sequential(
            ResidualBlock(initial_channels),
            ResidualBlock(initial_channels),
        )

        self.pixel_shuffle = nn.Sequential(
            PixelShuffleBlock(initial_channels, initial_channels // 2, upscale_factor=2),
            PixelShuffleBlock(initial_channels // 2, initial_channels // 4, upscale_factor=2),
            PixelShuffleBlock(initial_channels // 4, output_channels, upscale_factor=2)
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, input_dim)
        batch_size = x.size(0)

        # 全连接层将 x 变为 (batch_size, initial_channels * (H // 8) * (W // 8))
        x = self.fc(x)

        # 重塑为 (batch_size, initial_channels, H // 8, W // 8)
        x = x.view(batch_size, self.initial_channels, self.H // 8, self.W // 8)

        # 应用残差块
        x = self.res_block(x)

        # 应用 PixelShuffle 上采样
        x = self.pixel_shuffle(x)

        # 控制输出范围(-1. 1)
        x = self.tanh(x)

        return x


class DrawCanvas(nn.Module):
    """
    对Canvas进行Self-Attention进行画画,
    """

    def __init__(self, embed_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True,
                                                       dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=num_layers)

    def forward(self, x):
        canvas = self.encoder(x)
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
            [nn.ModuleList([
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(dropout)
            ]) for _ in range(num_layers)]
        )

    def forward(self, camera, blank_canvas):
        canvas = blank_canvas
        for attention, norm, dropout in self.layers:
            attended_canvas, _ = attention(canvas, camera, camera)
            canvas = canvas + dropout(attended_canvas)
            canvas = norm(canvas)
        return canvas


class Draw_Attention_Generator(nn.Module):
    def __init__(self, embed_dim: int, canvas_dim: int, H: int, W: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.canvas_dim = canvas_dim
        self.H = H
        self.W = W

        self.fc = nn.Linear(10, self.canvas_dim * self.embed_dim)
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.canvas_dim)
        self.canvas = Canvas(self.embed_dim, num_heads=32, dropout=0.2, num_layers=8)

        self.draw_canvas = DrawCanvas(self.embed_dim, num_heads=32, dropout=0.2, num_layers=8)
        self.canvas2picture = Canvas2Picture(H=self.H, W=self.W, input_dim=self.canvas_dim * self.embed_dim)

    def forward(self, camera, blank_canvas):
        batch_size = camera.size(0)

        camera = self.fc(camera)
        camera = camera.view(batch_size, self.canvas_dim, self.embed_dim)
        blank_canvas = blank_canvas.view(batch_size, self.canvas_dim, self.embed_dim)
        blank_canvas = self.positional_encoding(blank_canvas)
        canvas = self.canvas(camera, blank_canvas)
        canvas = self.draw_canvas(canvas)
        canvas = canvas.view(batch_size, -1)
        picture = self.canvas2picture(canvas)

        return picture


class Draw_Attention_Discriminator(nn.Module):
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

        )

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_path = './data/images'
    img_view_path = './data/inputs/images.csv'
    cameras_path = './data/inputs/cameras.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scale_factor = 0.125
    embed_dim = 32
    W = 3072
    H = 2304
    batch = 4
    canvas_dim = 32

    # Initialize generator and discriminator
    gen = Draw_Attention_Generator(embed_dim=embed_dim, canvas_dim=canvas_dim, H=288, W=384).to(device)
    dis = Draw_Attention_Discriminator(input_channels=3, feature_dim=64).to(device)
    dis.initialize_weights()

    # Generate random data
    blank_canvas = torch.randn((batch, canvas_dim * embed_dim), dtype=torch.float32).to(device)
    cameras = torch.randn(batch, 10).to(device)

    # Forward pass
    out = gen(cameras, blank_canvas)
    out2 = dis(out)

    # Print shapes of the outputs
    print("Generator output shape:", out.shape)
    print("Discriminator output shape:", out2.shape)

    # Initialize target and loss function
    valid = torch.ones(out2.shape).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Compute loss for discriminator and generator
    d_loss = criterion(out2, valid)

    # Initialize optimizers
    optimizer_G = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Zero the gradients
    optimizer_D.zero_grad()
    optimizer_G.zero_grad()

    # Backward pass for discriminator
    try:
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        print("Discriminator backward pass successful.")
    except RuntimeError as e:
        print(f"Error in Discriminator backward pass: {e}")

    # Compute loss for generator
    fake_labels = torch.ones(out2.shape).to(device)
    g_loss = criterion(dis(out.detach()), fake_labels)

    # Backward pass for generator
    try:
        g_loss.backward()
        optimizer_G.step()
        print("Generator backward pass successful.")
    except RuntimeError as e:
        print(f"Error in Generator backward pass: {e}")
