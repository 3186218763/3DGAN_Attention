import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm


def generate_ellipsoid(center, covariance, num_points=100):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            point = np.dot(np.diag(np.sqrt(eigenvalues)), point)
            point = np.dot(eigenvectors, point)
            x[i, j], y[i, j], z[i, j] = point + center

    return x, y, z


def compute_sh_color(l, m, theta, phi):
    sh_value = sph_harm(m, l, phi, theta).real
    r = 0.5 * (sh_value + 1)
    g = 0.5 * (sh_value + 1)
    b = 0.5 * (sh_value + 1)
    return r, g, b


def project_to_2d(x, y, z, img_size, R):
    x_rot, y_rot, z_rot = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            point_rot = np.dot(R, point)
            x_rot[i, j], y_rot[i, j], z_rot[i, j] = point_rot

    x_proj = x_rot
    y_proj = y_rot

    x_img = ((x_proj - x_proj.min()) / (x_proj.max() - x_proj.min()) * (img_size - 1)).astype(int)
    y_img = ((y_proj - y_proj.min()) / (y_proj.max() - y_proj.min()) * (img_size - 1)).astype(int)

    return x_img, y_img


# 定义高斯椭球参数
center1 = np.array([22.0, 11.0, 42.0])
covariance1 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

center2 = np.array([30.0, 15.0, 50.0])
covariance2 = np.array([
    [2.0, 0.5, 0.0],
    [0.5, 2.0, 0.0],
    [0.0, 0.0, 1.5]
])

# 定义旋转角度
angle_x, angle_y, angle_z = 30, 45, 60
angle_x, angle_y, angle_z = np.deg2rad([angle_x, angle_y, angle_z])

Rx = np.array([
    [1, 0, 0],
    [0, np.cos(angle_x), -np.sin(angle_x)],
    [0, np.sin(angle_x), np.cos(angle_x)]
])

Ry = np.array([
    [np.cos(angle_y), 0, np.sin(angle_y)],
    [0, 1, 0],
    [-np.sin(angle_y), 0, np.cos(angle_y)]
])

Rz = np.array([
    [np.cos(angle_z), -np.sin(angle_z), 0],
    [np.sin(angle_z), np.cos(angle_z), 0],
    [0, 0, 1]
])

R = np.dot(Rz, np.dot(Ry, Rx))

# 生成椭球
x1, y1, z1 = generate_ellipsoid(center1, covariance1)
x2, y2, z2 = generate_ellipsoid(center2, covariance2)

# 投影到2D平面
img_size = 512
x_img1, y_img1 = project_to_2d(x1, y1, z1, img_size, R)
x_img2, y_img2 = project_to_2d(x2, y2, z2, img_size, R)

# 计算球谐函数颜色值
theta1 = np.arccos(np.clip(z1, -1, 1))
phi1 = np.arctan2(y1, x1)
theta2 = np.arccos(np.clip(z2, -1, 1))
phi2 = np.arctan2(y2, x2)

l, m = 4, 1

colors1 = np.zeros((x1.shape[0], x1.shape[1], 3))
colors2 = np.zeros((x2.shape[0], x2.shape[1], 3))

for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        colors1[i, j] = compute_sh_color(l, m, theta1[i, j], phi1[i, j])
        colors2[i, j] = compute_sh_color(l, m, theta2[i, j], phi2[i, j])

# 创建白板
img = np.ones((img_size, img_size, 3))

# 在图像平面上绘制点并混合颜色
for i in range(x_img1.shape[0]):
    for j in range(x_img1.shape[1]):
        img[y_img1[i, j], x_img1[i, j]] = (img[y_img1[i, j], x_img1[i, j]] + colors1[i, j]) / 2

for i in range(x_img2.shape[0]):
    for j in range(x_img2.shape[1]):
        img[y_img2[i, j], x_img2[i, j]] = (img[y_img2[i, j], x_img2[i, j]] + colors2[i, j]) / 2

# 可视化投影图像
plt.imshow(img)
plt.title("Projected Gaussian Ellipsoids with Spherical Harmonics Colors")
plt.show()

