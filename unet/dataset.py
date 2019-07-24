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

from pre_processing import apply_transform, adjust_transform_for_image, normalize_image, TransformParameters

class SEMDataset(Dataset):
    def __init__(self, 
                image_dir, 
                label_dir, 
                transform_generator=None,
                transform_parameters=None,
                to_tensor=True):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (str): transform the input image
        """
        self.transform_generator    = transform_generator
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.to_tensor              = to_tensor

        self.image_paths = glob.glob(os.path.join(image_dir, "*.tiff"))
        self.image_paths.sort()

        self.label_paths = [os.path.join(label_dir, "%s.png" % os.path.basename(filepath).split(".")[0]) for filepath in self.image_paths]

    def _transform(self, image, label):
        image = normalize_image(image)

        # augment image and label
        if self.transform_generator is not None:
            transform   = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
            image       = apply_transform(transform, image, self.transform_parameters)
            label       = apply_transform(transform, label, self.transform_parameters)

        # Transform data to tensor
        if self.to_tensor:
            image = TF.to_tensor(image)
            # can not use to_tensor() for label 
            # since it will scale the data to range [0, 1]
            label = torch.from_numpy(label).long()

        return image, label

    def __getitem__(self, index):
        image = tiff.imread(self.image_paths[index])
        
        label = Image.open(self.label_paths[index]).convert('L')
        label = np.array(label)
        
        image, label = self._transform(image, label)

        return image, label

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    from transform import random_transform_generator
    transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
            )

    data_dir    = "D:\\Workspace\\data\\raw\\pi-sar2\\20110312\\patches"
    dataset     = SEMDataset(os.path.join(data_dir, "train", "img"), 
                        os.path.join(data_dir, "train", "label"), 
                        transform_generator=transform_generator, 
                        to_tensor=True)

    # check get item
    image, label = dataset.__getitem__(4)
    print(image.shape, label.shape)

    # tiff.imsave("image.tif", image, planarconfig='contig')
    # Image.fromarray(label).save("label.png")