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
from params import *

class SEMDataset(Dataset):

    def __init__(self, image_dir, label_dir, stage="train"):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        self.stage = stage

        self.image_paths = glob.glob(os.path.join(image_dir, "*.tiff"))
        self.image_paths.sort()

        if self.stage in ["train", "val"]:
            self.label_paths = glob.glob(os.path.join(label_dir, "*.png"))
            self.label_paths.sort()

    def transform(self, image, label=None):
        if self.stage == "train":
            # Resize
            resize = transforms.Resize(size=(520, 520))
            label = resize(label)

            i, j, h, w = transforms.RandomCrop.get_params(label, output_size=(512, 512))
            label = TF.crop(label, i, j, h, w)

            flip_horizontal = random.random() > 0.5
            if flip_horizontal:
                label = TF.hflip(label)

            flip_vertical = random.random() > 0.5
            if flip_vertical:
                label = TF.vflip(label)
            
            transform_img = np.zeros((image.shape[2], image.shape[0], image.shape[1]))            
            for c in range(image.shape[2]):
                slide = Image.fromarray(image[..., c])
                slide = resize(slide)
                slide = TF.crop(slide, i, j, h, w)
                if flip_horizontal:
                    slide = TF.hflip(slide)
                if flip_vertical:
                    slide = TF.vflip(slide)

                transform_img[c, :, :] = np.array(slide)

            label = np.array(label)
            return torch.from_numpy(transform_img).float(), torch.from_numpy(label).long()

        # Transform to tensor
        image = TF.to_tensor(image)
        if label is not None:
            label = np.array(label)
            label = torch.from_numpy(label).long()

        return image, label

    def __getitem__(self, index):
        image = tiff.imread(self.image_paths[index]).astype(np.uint8)

        if self.stage in ["train", "val"]:
            label = Image.open(self.label_paths[index]).convert('L')
            image, label = self.transform(image, label)

            return image, label
        else:
            no_data_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
            no_data_mask[image[:, :, 0] == -99] = 0

            image, _ = self.transform(image)

            return image, no_data_mask, os.path.basename(self.image_paths[index]).split(".")[0]

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    SEM_train = SEMDataset(os.path.join(data_dir, "train", "img"), os.path.join(data_dir, "train", "mask"), stage="train")
    
    img, msk = SEM_train.__getitem__(1000)
    print(img)