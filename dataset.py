from torch.utils.data import Dataset
from utools import get_samples


class GS_Dataset(Dataset):
    """
    sample = {
            "camera_params": camera_params,
            "label": img,
        }
    """

    def __init__(self, cameras_path, img_view_path, label_path, transform=None, scale_factor=1.):
        self.transform = transform
        self.samples = get_samples(label_path, img_view_path, cameras_path, scale_factor=scale_factor)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        camera_params = sample['camera_params']
        label = sample['label']
        if self.transform is not None:
            label = self.transform(label)

        return camera_params, label

    def __len__(self):
        return len(self.samples)
