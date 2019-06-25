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
    def __init__(self, image_dir, label_dir, stage="train", n_batches_factor=2 ):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        self.stage = stage
        self.n_batches_factor = n_batches_factor

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
            flip_horizontal = np.random.rand() > 0.5
            if flip_horizontal:
                label = TF.hflip(label)
            flip_vertical = np.random.rand() > 0.5
            if flip_vertical:
                label = TF.vflip(label)
            
            transform_img = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)            
            for c in range(image.shape[2]):
                slide = Image.fromarray(image[..., c])
                slide = resize(slide)
                slide = TF.crop(slide, i, j, h, w)
                if flip_horizontal:
                    slide = TF.hflip(slide)
                if flip_vertical:
                    slide = TF.vflip(slide)

                transform_img[:, :, c] = np.array(slide)

            label = np.array(label)

            return TF.to_tensor(transform_img), torch.from_numpy(label).long()
        else:
            # Transform to tensor
            image = TF.to_tensor(image)
            if label is not None:
                label = np.array(label)
                label = torch.from_numpy(label).long()

            return image, label

    def __getitem__(self, index):
        index = index // self.n_batches_factor

        image = tiff.imread(self.image_paths[index]).astype(np.uint8)
        if self.stage in ["train", "val"]:
            label = Image.open(self.label_paths[index]).convert('L')
            image, label = self.transform(image, label)

            return image, label
        else:
            no_data_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
            no_data_mask[image[:, :, 0] == 0] = 0

            image, _ = self.transform(image)

            return image, no_data_mask, os.path.basename(self.image_paths[index])[:-5] #.split(".")[0]

    def __len__(self):
        return len(self.image_paths) * self.n_batches_factor

if __name__ == "__main__":
    dataset = SEMDataset(os.path.join(data_dir, "val", "img"), os.path.join(data_dir, "val", "label"), stage="train")

    # check get item
    image, label = dataset.__getitem__(4)
    tiff.imsave("image.tif", image, planarconfig='contig')
    Image.fromarray(label).save("label.png")