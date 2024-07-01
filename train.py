from torch.utils.data import DataLoader
from dataset import GS_Dataset
from net import Attention_3DGS
from torch import optim
import torch
import torch.nn as nn
from tqdm import tqdm

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
    num_epochs = 1
    net = Attention_3DGS(input_dim=input_dim,
                         seq_len=seq_len,
                         embed_dim=embed_dim,
                         num_points=num_points,
                         max_val=max_val,
                         min_val=min_val, ).to(device)
    dataset = GS_Dataset(img_path='./data/images', data_path='./data/inputs/images.csv')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    for epoch in range(num_epochs):
        for batch_idx, (data, img) in enumerate(dataloader):
            data = data.to(device)
            img = img.to(device)
            optimizer.zero_grad()
            output = net(data)
            print(output.shape)
            loss = loss_func(output, img)
            loss.backward()
            optimizer.step()




