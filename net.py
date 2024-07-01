import torch
import torch.nn as nn


class Self_Attention_Block(nn.Module):
    def __init__(self,
                 seq_len,
                 input_dim,
                 embed_dim,
                 num_layers=2,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, embed_dim),
        )
        self.norm = nn.LayerNorm([seq_len, embed_dim])
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.norm(self.fc(x))

        for _, layer in enumerate(self.layers):
            x, _ = layer(x, x, x)
            x = self.norm(x)

        return x


class Ellipsoids(nn.Module):
    def __init__(self,
                 num_points,
                 matrix_dim,
                 min_val,
                 max_val,
                 rotate_dim=64,
                 scale_dim=64,
                 ):
        super().__init__()

        # 定义模型参数
        self.Center = nn.Parameter(torch.empty(num_points, matrix_dim).uniform_(min_val, max_val))
        self.Rotate = nn.Parameter(torch.ones(num_points, rotate_dim))
        self.Scale = nn.Parameter(torch.ones(num_points, scale_dim))

    def forward(self, x):
        # 拼接模型参数
        gaussian_ellipsoids = torch.cat((self.Center, self.Rotate, self.Scale), dim=1)

        # 将拼接后的张量移到输入 x 所在的设备上
        gaussian_ellipsoids = gaussian_ellipsoids.to(x.device)

        return gaussian_ellipsoids


class Cross_Attention_Block(nn.Module):
    def __init__(self,
                 seq_len,
                 num_points,
                 embed_dim,
                 num_layers,
                 num_heads,
                 dropout):
        super().__init__()
        self.norm_paper = nn.LayerNorm([seq_len, embed_dim])
        self.norm_brush = nn.LayerNorm([num_points, embed_dim])
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout, add_bias_kv=True)
             for _ in range(num_layers)]
        )

    def forward(self, q, kv):
        for _, layer in enumerate(self.layers):
            q, _ = layer(q, kv, kv)
        return q


class Attention_3DGS(nn.Module):
    def __init__(self,
                 input_dim,
                 seq_len,
                 embed_dim,
                 num_points,
                 matrix_dim=3,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 min_val=-10.,
                 max_val=10.,
                 rotate_dim=64,
                 scale_dim=64,
                 paper_size=(512, 512, 3)):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.H, self.W, self.C = paper_size
        self.fc_seq = nn.Linear(input_dim, input_dim * seq_len)
        self.gen_gaussian_ellipsoids = Ellipsoids(num_points, matrix_dim, min_val, max_val, rotate_dim, scale_dim)
        self.paper_self_attention = Self_Attention_Block(seq_len, input_dim, embed_dim)
        self.brush_self_attention = Self_Attention_Block(num_points, matrix_dim + rotate_dim + scale_dim, embed_dim)
        self.draw_cross_attention = Cross_Attention_Block(seq_len, num_points, embed_dim, num_layers, num_heads,
                                                          dropout)
        self.get_drawing = nn.Sequential(
            nn.Unflatten(1, (self.H, self.W, self.C)),
            nn.ReLU(),
            nn.Hardtanh(-1, 1)
        )

    def forward(self, x):
        x = self.fc_seq(x)  # (batch, 6) -> (batch, seq_len*input_dim)
        q = x.view(-1, self.seq_len, self.input_dim)  # (batch, seq_len*6) -> (batch, seq_len, 6)
        kv = self.gen_gaussian_ellipsoids(q).unsqueeze(0).expand(x.size(0), -1, -1)  # (num_points, 9)->(batch,
        # num_points, 9)
        paper_q = self.paper_self_attention(q)  # (batch, seq_len, embed_dim)
        brush_kv = self.brush_self_attention(kv)  # (batch, num_points, embed_dim)
        x = self.draw_cross_attention(paper_q, brush_kv)  # (batch, seq_len, embed_dim)
        x = x.reshape(-1, self.seq_len * self.embed_dim)  # (batch, seq_len, embed_dim) -> (batch, seq_len*embed_dim)
        x = self.get_drawing(x)
        x = (x+1.)*127.5
        x = x.round()

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 暂时不要改变
    embed_dim = 1024
    seq_len = 768
    input_dim = 6

    # 可以调整的数
    num_points = 1024
    batch_size = 8
    max_val = 1.
    min_val = -1.

    tensor = torch.randn(batch_size, input_dim).to(device)

    net = Attention_3DGS(input_dim=input_dim,
                         seq_len=seq_len,
                         embed_dim=embed_dim,
                         num_points=num_points,
                         max_val=max_val,
                         min_val=min_val, ).to(device)
    paper = net(tensor)

    print(paper[0])
