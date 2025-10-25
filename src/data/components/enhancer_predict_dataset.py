import os

import torch
import wsq
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class EnhancerPredictionDataset(Dataset):
    def __init__(self, data_dir: str = "data/", data_list: str = None, transform=None, img_subdir = '/latents/'):
        self.data_dir        = data_dir
        self.transform       = transform
        self.data_list       = data_dir + data_list if data_list is not None else data_dir + "/data_list.txt"

        with open(self.data_list) as fp:
            lines = fp.readlines()

        self.data = [line.strip() for line in lines]

        self.img_suffix   = "." + os.listdir(data_dir + img_subdir)[0].split(".")[-1]

        self.img_subdir   = img_subdir



    def __getitem__(self, ix):
        img   = Image.open(self.data_dir + self.img_subdir   + self.data[ix] + self.img_suffix)

        # normalizing lat and ref to -1, 1

        img = transforms.ToTensor()(img)


        img_mean = torch.mean(img)
        img_std  = torch.std(img)


        img = transforms.Normalize(mean=[img_mean], std=[2 * img_std])(img)

        # print(self.data[ix])

        return img, self.data[ix]

    def __len__(self):
        return len(self.data)
