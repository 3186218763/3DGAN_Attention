from torch.utils.data import Dataset
from utools import get_samples


class GS_Dataset(Dataset):
    """
    sample = {
            'data': data,
            'img': img,
        }
    """
    def __init__(self, img_path="./data/images", data_path="./data/inputs/images.csv"):
        self.samples = get_samples(img_path=img_path, data_path=data_path)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample

    def __len__(self):
        return len(self.samples)
