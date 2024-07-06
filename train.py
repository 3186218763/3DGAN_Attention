from torch.utils.data import DataLoader
from dataset import GS_Dataset
from net import Draw_Attention_Generator, Draw_Attention_Discriminator
from torch import optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms


def train():
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_path = './data/images'
    img_view_path = './data/inputs/images.csv'
    cameras_path = './data/inputs/cameras.csv'
    gen_model_path = None
    num_epochs = 1000000
    scale_factor = 0.125
    embed_dim = 32
    W = int(3072 * scale_factor)
    H = int(2304 * scale_factor)
    batch = 4
    num_reduction = 12
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Resize((H, W)),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = GS_Dataset(cameras_path, img_view_path, label_path, transform=transform, scale_factor=scale_factor)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4)

    gen = Draw_Attention_Generator(embed_dim=embed_dim, H=H, W=W,
                                   num_reduction=num_reduction).to(device)  # (384, 288)
    if gen_model_path is not None:
        gen.load_state_dict(torch.load(gen_model_path))
        print("生成器参数加载成功")

    dis = Draw_Attention_Discriminator(feature_dim=64).to(device)
    loss_fn = nn.BCELoss()
    optimizer_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_dis = optim.Adam(dis.parameters(), lr=1e-5, betas=(0.5, 0.999))
    scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen, step_size=50, gamma=0.7)
    scheduler_dis = optim.lr_scheduler.StepLR(optimizer_dis, step_size=50, gamma=0.7)
    size = (batch, H, W, embed_dim)
    for epoch in range(num_epochs):
        gen.train()
        dis.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # 获取数据和标签
            camera_params, label = batch
            camera_params, label = camera_params.to(device), label.to(device)
            blank_canvas = torch.randn(size=size, dtype=torch.float32, device=device)

            # 训练生成器
            optimizer_gen.zero_grad()

            gen_imgs = gen(camera_params, blank_canvas)
            gen_dis = dis(gen_imgs)
            valid = torch.ones(gen_dis.shape).to(device)
            fake = torch.zeros(gen_dis.shape).to(device)

            g_loss = loss_fn(gen_dis, valid)

            g_loss.backward()
            optimizer_gen.step()

            optimizer_dis.zero_grad()
            real_loss = loss_fn(dis(gen_imgs.detach()), valid)
            fake_loss = loss_fn(dis(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_dis.step()

            # 更新进度条
            progress_bar.set_postfix({"g_loss": g_loss.item(), "d_loss": d_loss.item()})

        # 更新学习率
        scheduler_gen.step()
        scheduler_dis.step()

        if epoch % 50 == 0:
            torch.save(gen.state_dict(), './pth/generator.pth')


if __name__ == '__main__':
    train()
