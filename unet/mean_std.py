import numpy as np

import os

import torch
from torch.utils.data import DataLoader

from dataset import SEMDataset

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion
        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = None
    snd_moment = None

    for (data, _) in loader:
        b, h, w, c = data.shape

        if fst_moment is None:
            fst_moment = torch.empty(c)
        if snd_moment is None:
            snd_moment = torch.empty(c)

        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 1, 2])
        sum_of_square = torch.sum(data ** 2, dim=[0, 1, 2])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

if __name__ == '__main__':
    data_dir    = "D:\\Workspace\\data\\raw\\pi-sar2\\20110312\\patches"
    dataset     = SEMDataset(os.path.join(data_dir, "train", "img"), 
                        os.path.join(data_dir, "train", "label"), 
                        transform_generator=None, 
                        to_tensor=False)
    loader    = DataLoader(dataset=dataset, num_workers=0, batch_size=4, shuffle=False)

    mean, std = online_mean_and_sd(loader)
    print(mean, std)