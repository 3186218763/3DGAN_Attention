import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import numpy as np

transform = transforms.Compose([
    transforms.Resize((400, 500)),
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize(0.5, 0.5)
])


def get_cameras_csv(cameras_txt):
    cameras = pd.read_csv(cameras_txt)
    with open(cameras_txt, 'r') as file:
        lines = file.readlines()

    # 提取相机数据
    camera_data = []
    for line in lines:
        if not line.startswith('#') and line.strip():
            parts = line.strip().split()
            camera_id = parts[0]
            model = parts[1]
            width = parts[2]
            height = parts[3]
            params = parts[4:]  # 剩下的部分是内参
            camera_data.append([camera_id, model, width, height] + params)

    # 创建DataFrame
    df = pd.DataFrame(camera_data,
                      columns=['CAMERA_ID', 'MODEL', 'W', 'H', 'FOCAL_LENGTH', 'PRINCIPAL_POINT_X',
                               'PRINCIPAL_POINT_Y', 'DISTORTION'])

    # 保存为csv文件
    csv_file = './data/inputs/cameras.csv'
    df.to_csv(csv_file, index=False)


def get_points(num_points=50000, points_csv_path='../data/inputs/points3D.csv'):
    df = pd.read_csv(points_csv_path)
    xyz_data = df[['X', 'Y', 'Z']].values
    rgb_data = df[['R', 'G', 'B']].values
    centers = np.array(xyz_data[:num_points, :])
    rgbs = rgb_data[:num_points, :]

    disturbance = np.random.normal(0, 0.01, centers.shape)
    centers = centers + disturbance
    rgbs = rgbs + disturbance
    return centers, rgbs


def get_points_csv(points_path='./data/sparse/points3D.txt'):
    data = []

    # Read lines excluding comments
    with open(points_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                parts = line.strip().split()
                point_id = int(parts[0])
                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])
                data.append([x, y, z, r, g, b])

    # Select even-indexed rows and create DataFrame
    columns = ['X', 'Y', 'Z', 'R', 'G', 'B']
    df = pd.DataFrame(data, columns=columns)
    df.index.name = 'IDX'

    # 保存到新的CSV文件
    df.to_csv('./data/inputs/points3D.csv')


def get_images_csv(images_path="./data/sparse/images.txt"):
    data = []

    # Read lines excluding comments
    with open(images_path, 'r') as file:
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


def get_samples(label_path, img_view_path, cameras_path, scale_factor=1.):
    df = pd.read_csv(img_view_path, low_memory=False)
    ca_df = pd.read_csv(cameras_path, low_memory=False)
    focal_length = float(ca_df.iloc[0]['FOCAL_LENGTH']) * scale_factor
    principal_point_x = float(ca_df.iloc[0]['PRINCIPAL_POINT_X']) * scale_factor
    principal_point_y = float(ca_df.iloc[0]['PRINCIPAL_POINT_Y']) * scale_factor
    distortion = float(ca_df.iloc[0]['DISTORTION'])

    # 将内参转换为numpy数组，然后转换为Tensor
    intrinsics = np.array([focal_length, principal_point_x, principal_point_y, distortion])
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    samples = []
    for _, row in df.iterrows():
        quaternion_cols = ['QW', 'QX', 'QY', 'QZ']
        translation_cols = ['TX', 'TY', 'TZ']
        quaternion = row[quaternion_cols].values.astype(float)
        translation = row[translation_cols].values.astype(float)

        rotation_matrix = R.from_quat(quaternion).as_matrix()
        rotation_world = rotation_matrix.T
        translation_world = -rotation_matrix.T @ np.array(translation)

        # 相机外参（假设是某个时间点的外参）
        camera_position = translation_world
        camera_orientation = rotation_world[:, 2] / np.linalg.norm(rotation_world[:, 2])
        extrinsics = np.hstack((camera_position, camera_orientation))
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)

        camera_params = torch.cat([intrinsics, extrinsics])
        name = row['NAME']
        img = Image.open(os.path.join(label_path, name))
        sample = {
            "camera_params": camera_params,
            "label": img,
        }
        samples.append(sample)
    return samples


if __name__ == '__main__':
    label_path = './data/images'
    img_view_path = './data/inputs/images.csv'
    cameras_path = './data/inputs/cameras.csv'
    samples = get_samples(label_path, img_view_path, cameras_path)
    sample = samples[0]
    print(sample["label"].shape)
