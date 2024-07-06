from net import Draw_Attention_Generator
from utools import tensor_to_image, save_image
import torch


def predict(gen, camera_params, blank_canvas):
    img_tensor = gen(camera_params, blank_canvas)
    img = img_tensor.detach().cpu().numpy()
    img = tensor_to_image(img)
    save_image(img)


if __name__ == '__main__':
    scale_factor = 0.125
    embed_dim = 32
    W = int(3072 * scale_factor)
    H = int(2304 * scale_factor)

    canvas_dim = 32
    gen_model_path = 'arg_load/gen.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Draw_Attention_Generator(embed_dim=embed_dim, canvas_dim=canvas_dim, H=H, W=W).to(device)
    if gen_model_path is not None:
        gen.load_state_dict(torch.load(gen_model_path))
        print("生成器参数加载成功")
    gen.eval()
    values = [2559.7, 1536.0, 1152.0, -0.0205, 4.7543, -0.0383, -1.3849, -0.9062, 0.2739, 0.3221]
    camera_params = torch.tensor(values, dtype=torch.float32, device=device)
    camera_params = camera_params.unsqueeze(0)
    size = (1, canvas_dim*embed_dim)
    blank_canvas = torch.randn(size=size, dtype=torch.float32, device=device)
    predict(gen, camera_params, blank_canvas)
