import argparse, sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from advanced_model import UNet
from dataset import *
from modules import *
from save_history import *
from params import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Multualy exclusive with --predict.', action='store_true')
parser.add_argument('--predict', help='Multualy exclusive with --train.', action='store_true')
parser.add_argument('--model_name', type=str)
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
    # Dataset begin
    train_dataset = SEMDataset(os.path.join(data_dir, "train", "img"), os.path.join(data_dir, "train", "label"), stage="train")
    val_dataset = SEMDataset(os.path.join(data_dir, "val", "img"), os.path.join(data_dir, "val", "label"), stage="val")
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    
    save_file_name = os.path.join(save_dir, "history.csv")

    rate_schedule = np.ones(args.n_epoch)
    # rate_schedule[:args.n_gradual] = np.linspace(1.0, args.keep_rate, args.n_gradual)

    for i in range(args.n_epoch):
        # train the model
        train_model(model, train_loader, criterion, optimizer, scheduler, ignore_index=0, keep_rate=rate_schedule[i])
        # Validation every 5 epoch
        if (i + 1) % args.val_interval == 0:
            train_loss = get_loss(model, train_loader, criterion)
            val_loss = get_loss(model, val_loader, criterion)
            
            print('Epoch %d, Train loss: %.3f, Val loss: %.3f' % (i + 1, train_loss, val_loss))

            values = [i + 1, train_loss, val_loss]
            export_history(header, values, save_dir, save_file_name)

        if (i + 1) % args.save_interval == 0:  # save model every 10 epoch
            save_models(model, save_dir, i + 1)

def predict():
    test_dataset = SEMDataset(os.path.join(data_dir, "train", "img"), None, stage="test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    model = torch.load(os.path.join(save_dir, args.model_name))
    model = model.cuda()

    model.eval()
    pbar = tqdm(test_loader)
    for batch_idx, (images, masks, basenames) in enumerate(pbar):
        images = images.cuda()
        probs = F.softmax(model.forward(images)).data.cpu().numpy()
        preds = np.argmax(probs, axis=1).astype(np.uint8)

        masks = masks.numpy()
        for pred, mask, basename in zip(preds, masks, basenames):
            pred = pred * mask

            # color_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            # (color_img[:, :, 0])[pred == 1] = 128
            # (color_img[:, :, 1])[pred == 2] = 128
            # color_img = Image.fromarray(color_img)
            # color_img.save(os.path.join(save_dir, "visualizations", "%s_c.png" % basename))

            label = Image.fromarray(pred)
            label.save(os.path.join(save_dir, "predictions", "%s.png" % basename))
            
if __name__ == "__main__":
    if args.train:
        train()
    elif args.predict:
        predict()
    else:
        print("Please chose --train, --predict. Mutualy exclusive!")