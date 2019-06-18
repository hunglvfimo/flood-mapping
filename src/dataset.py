import os
import glob
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import tifffile as tiff

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable

from pre_processing import *

data_dir = os.path.join("..", "..", "..", "data", "processed", "PI_SAR2_FINE")

class SEMDataset(Dataset):

    def __init__(self, image_path, mask_path, is_train=False):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        # all file names
        self.image_paths = glob.glob(os.path.join(mask_path, "*"))
        self.image_paths.sort()

        self.mask_paths = glob.glob(os.path.join(image_path, "*"))
        self.mask_paths.sort()

        self.is_train = is_train

    def transform(self, image, mask):
        if self.is_train:
            # Resize
            resize = transforms.Resize(size=(520, 520))
            image = resize(image)
            mask = resize(mask)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __getitem__(self, index):
        image = tiff.imread(self.image_paths[index])

        mask = Image.open(self.mask_paths[index]).convert('L')
        
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    SEM_train = SEMDataset(os.path.join(data_dir, "train", "img"), os.path.join(data_dir, "train", "mask"), is_train=True)
    
    img, msk = SEM_train.__getitem__(1000)
    print(img)