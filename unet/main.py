import argparse, sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import UNet
from transform import random_transform_generator
from dataset import SEMDataset
from modules import *
from save_history import *
from params import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Multualy exclusive with --predict.', action='store_true')
parser.add_argument('--predict', help='Multualy exclusive with --train.', action='store_true')
parser.add_argument('--snapshot', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--keep_rate', type = float, default=0.5)
parser.add_argument('--n_gradual', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--save_interval', type=int, default=10)

args = parser.parse_args()

def train():
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

    # Dataset begin
    train_dataset   = SEMDataset(os.path.join(data_dir, "train", "img"), os.path.join(data_dir, "train", "label"), stage="train")
    val_dataset     = SEMDataset(os.path.join(data_dir, "val", "img"), os.path.join(data_dir, "val", "label"), stage="val")
    # Dataset end

    # Dataloader begins
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader      = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    # Dataloader end

    # Model
    model = UNet(n_class=3)
    if args.snapshot:
        model = torch.load(args.snapshot)
    model = model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Optimizerd
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    
    save_file_name = os.path.join(args.save_dir, "history.csv")

    rate_schedule = np.ones(args.n_epoch)
    for i in range(args.n_epoch):
        # train the model
        train_model(model, train_loader, criterion, optimizer, scheduler, keep_rate=rate_schedule[i])
        
        # Validation every 5 epoch
        if (i + 1) % args.val_interval == 0:
            train_loss  = get_loss(model, train_loader, criterion)
            val_loss    = get_loss(model, val_loader, criterion)
            
            print('Epoch %d, Train loss: %.3f, Val loss: %.3f' % (i + 1, train_loss, val_loss))

            values = [i + 1, train_loss, val_loss]
            export_history(header, values, args.save_dir, save_file_name)

        if (i + 1) % args.save_interval == 0:  # save model every 10 epoch
            save_models(model, args.save_dir, i + 1)

def predict():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_dataset = SEMDataset(args.data_dir, None, stage="test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.snapshot)
    model = model.cuda()

    model.eval()
    pbar = tqdm(test_loader)
    for batch_idx, (images, novalue_masks, basenames) in enumerate(pbar):
        images = images.cuda()
        
        probs = F.softmax(model.forward(images)).data.cpu().numpy()
        
        preds = np.argmax(probs, axis=1).astype(np.uint8)

        lowprob_masks = np.max(probs, axis=1)
        lowprob_masks = (lowprob_masks > 0.9).astype(np.uint8)

        novalue_masks = novalue_masks.numpy()
        
        for pred, lowprob_mask, novalue_mask, basename in zip(preds, lowprob_masks, novalue_masks, basenames):
            pred = pred * novalue_mask * lowprob_mask

            label = Image.fromarray(pred)
            label.save(os.path.join(args.save_dir, "%s.png" % basename))
            
if __name__ == "__main__":
    if args.train:
        train()
    elif args.predict:
        predict()
    else:
        print("Please chose --train, --predict. Mutualy exclusive!")