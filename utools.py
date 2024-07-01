import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import numpy as np

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize image to 512x512
    transforms.ToTensor()  # Convert PIL Image to Tensor
])


def get_samples(img_path, data_path):
    df = pd.read_csv(data_path, low_memory=False)

    samples = []
    for _, row in df.iterrows():
        quaternion_cols = ['QW', 'QX', 'QY', 'QZ']
        translation_cols = ['TX', 'TY', 'TZ']
        quaternion = row[quaternion_cols].values.astype(float)
        translation = row[translation_cols].values.astype(float)

        rotation_matrix = R.from_quat(quaternion).as_matrix()
        rotation_world = rotation_matrix.T
        translation_world = -rotation_matrix.T @ np.array(translation)

        camera_position_world = translation_world
        camera_direction_world = rotation_world[:, 2] / np.linalg.norm(rotation_world[:, 2])

        data = np.hstack((camera_position_world, camera_direction_world))
        data = torch.tensor(data).float()

        name = row['NAME']
        img = Image.open(os.path.join(img_path, name))  # Assuming name is the image filename
        img = transform(img)
        img = img.permute(1, 2, 0)
        sample = {
            'data': data,
            'img': img,
        }
        samples.append(sample)
    return samples


def get_input_data(txt_path="./data/sparse/images.txt"):
    data = []

    # Read lines excluding comments
    with open(txt_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                data.append(line.strip().split())

    # Select even-indexed rows and create DataFrame
    selected_data = [data[i] for i in range(0, len(data), 2)]
    df = pd.DataFrame(selected_data,
                      columns=['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME'])

    # Set NAME as index
    df.set_index('NAME', inplace=True)

    save_path = './data/inputs'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df.to_csv(os.path.join(save_path, 'images.csv'), index=True, header=True)


if __name__ == '__main__':
    get_input_data()

