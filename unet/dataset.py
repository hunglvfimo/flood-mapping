import os
import glob

import numpy as np
import random

from PIL import Image
import tifffile as tiff

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

from pre_processing import apply_transform
from params import *

class SEMDataset(Dataset):
    def __init__(self, 
                image_dir, 
                label_dir, 
                transform_generator=None,
                transform_parameters=None):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (str): transform the input image
        """
        self.transform_generator    = transform_generator
        self.transform_parameters   = transform_parameters or TransformParameters()

        self.image_paths = glob.glob(os.path.join(image_dir, "*.tiff"))
        self.image_paths.sort()

        self.label_paths = [os.path.join(label_dir, "%s.png" % os.path.basename(filepath).split(".")[0]) for filepath in self.image_paths]

    def transform(self, image, label):
        image = normalize_image(image)

        if self.stage == "train":
            # augment image and label
            if self.transform_generator is not None:
                image = apply_transform(self.transform_generator, image, self.transform_parameters)

        # Transform data to tensor
        image = TF.to_tensor(image)
        # can not use to_tensor() for label 
        # since it will scale the data to range [0, 1]
        label = torch.from_numpy(label).long()

        return image, label

    def __getitem__(self, index):
        image = tiff.imread(self.image_paths[index])
        
        label = Image.open(self.label_paths[index]).convert('L')
        label = np.array(label)
        
        image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    dataset = SEMDataset(os.path.join(data_dir, "val", "img"), os.path.join(data_dir, "val", "label"), stage="train")

    # check get item
    image, label = dataset.__getitem__(4)
    tiff.imsave("image.tif", image, planarconfig='contig')
    Image.fromarray(label).save("label.png")