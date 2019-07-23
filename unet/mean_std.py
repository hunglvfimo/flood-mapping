import numpy as np
from PIL import Image
import glob

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion
        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for (data, _) in loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def cal_dataset_stats():
    ds = ImageFolder(os.path.join(DATA_DIR, args.dataset, "train"),
                    transform=transforms.Compose([
                        transforms.Resize((args.input_size, args.input_size)),
                        transforms.ToTensor(),
                    ]))
    print(ds.class_to_idx)

    data_loader = DataLoader(ds, batch_size=16, shuffle=False, **kwargs)

    mean, std = online_mean_and_sd(data_loader)
    print(mean, std)