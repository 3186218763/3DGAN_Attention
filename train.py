from torch.utils.data import DataLoader
from dataset import GS_Dataset
from net import Attention_3DGS
from torch import optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    num_epochs = 100000
    # 定义模型、数据集、数据加载器、优化器和调度器
    net = Attention_3DGS(input_dim=input_dim,
                         seq_len=seq_len,
                         embed_dim=embed_dim,
                         num_points=num_points,
                         max_val=max_val,
                         min_val=min_val).to(device)
    dataset = GS_Dataset(img_path='./data/images', data_path='./data/inputs/images.csv')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    loss_func = nn.MSELoss()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0  # 初始化每个 epoch 的损失
        # 创建进度条
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (data, img) in pbar:
            data = data.to(device)
            img = img.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = loss_func(output, img)
            loss.backward()
            optimizer.step()

            # 累积损失
            epoch_loss += loss.item()

            # 更新进度条的描述信息
            pbar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 记录每个 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)

        # 更新学习率调度器
        scheduler.step()

        # 每 10 个 epoch 绘制一次损失曲线
        if (epoch + 1) % 100 == 0:
            plt.figure()
            plt.plot(loss_history, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss up to Epoch {epoch + 1}')
            plt.legend()
            plt.savefig(f'./pth/loss.png')
            torch.save(net.state_dict(), f'./pth/model.pth')
