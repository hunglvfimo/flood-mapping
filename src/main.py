import argparse, sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable

from advanced_model import UNet
from dataset import *
from modules import *
from save_history import *

data_dir = os.path.join("..", "..", "..", "data", "processed", "PI_SAR2_FINE")
save_dir = os.path.join("..", "..", "..", "models")

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--save_interval', type=int, default=10)

args = parser.parse_args()

if __name__ == "__main__":
    # Dataset begin
    train_dataset = SEMDataset(os.path.join(data_dir, "train", "img"), os.path.join(data_dir, "train", "mask"), is_train=True)
    val_dataset = SEMDataset(os.path.join(data_dir, "val", "img"), os.path.join(data_dir, "val", "mask"), is_train=False)
    # Dataset end

    # Dataloader begins
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    # Dataloader end

    # Model
    model = UNet(n_class=3)
    model = model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Optimizerd
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    
    save_file_name = os.path.join(save_dir, "history.csv")

    # Train
    print("Initializing Training!")
    for i in range(args.epoch_start, args.n_epoch):
        # train the model
        train_model(model, train_loader, criterion, optimizer)
        # Validation every 5 epoch
        if (i + 1) % args.val_interval == 0:
            train_acc, train_loss = get_loss(model, train_loader, criterion)
            val_acc, val_loss = get_loss(model, val_loader, criterion)
            
            print('Epoch %d, Train loss: %.3f, Train acc: %.3f, Val loss: %.3f, Val acc: %.3f' % (i + 1, train_loss, train_acc, val_loss, val_acc))

            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

        if (i + 1) % args.save_interval == 0:  # save model every 10 epoch
            save_models(model, save_dir, i + 1)