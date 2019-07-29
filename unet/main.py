import argparse, sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from transform import random_transform_generator
from dataset import SEMDataset
from modules import *
from save_history import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Multualy exclusive with --predict.', action='store_true')
parser.add_argument('--predict', help='Multualy exclusive with --train.', action='store_true')
parser.add_argument('--transform', help='', action='store_true')
parser.add_argument('--model_depth', type=int, default=5)
parser.add_argument('--snapshot', type=str)
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--init_epoch', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--save_interval', type=int, default=10)


args = parser.parse_args()

def train():
    # transform generator
    if args.transform:
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
    else:
        transform_generator = None

    # create custome dataset
    train_dataset   = SEMDataset(os.path.join(args.train_dir, "img"), 
                            os.path.join(args.train_dir, "label"), 
                            transform_generator=transform_generator)
    val_dataset     = SEMDataset(os.path.join(args.val_dir, "img"), 
                            os.path.join(args.val_dir, "label"))

    # Dataloader
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader      = torch.utils.data.DataLoader(dataset=val_dataset,   num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # Model
    # from advance_model import UNet
    # model = UNet(in_channels=11, n_classes=3, depth=args.model_depth, batch_norm=True, padding=True)
    
    from model import UNet
    model = UNet(in_channels=11, n_classes=2)

    if args.snapshot:
        model = torch.load(args.snapshot)
    model = model.cuda()

    # Loss function
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion   = nn.BCEWithLogitsLoss(reduction='none')

    # Optimizerd
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # Saving History to csv
    header = ['epoch', 'train_loss', 'val_acc']
    
    save_file_name = os.path.join(args.save_dir, "history.csv")

    for i in range(args.init_epoch, args.init_epoch + args.n_epoch):
        # train the model
        train_loss = train_model(model, train_loader, criterion, optimizer, scheduler)
        
        # validation every 5 epoch
        if (i + 1) % args.val_interval == 0:
            val_acc = evaluate_model(model, val_loader, criterion, metric=True)
            print('Epoch %d, Train loss: %.5f, Val acc: %.4f' % (i + 1, train_loss, val_acc))

            values  = [i + 1, train_loss, val_acc]
            export_history(header, values, args.save_dir, save_file_name)

        if (i + 1) % args.save_interval == 0:  # save model every save_interval epoch
            save_models(model, args.save_dir, i + 1)

def predict():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset     = SEMDataset(os.path.join(args.val_dir, "img"), 
                        os.path.join(args.val_dir, "label"), 
                        transform_generator=None)
    loader      = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.snapshot)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.cuda()
            labels = labels.numpy()
            
            probs           = F.softmax(model.forward(images)).data.cpu().numpy()
            
            preds           = np.argmax(probs, axis=1).astype(np.uint8)
            probs           = np.max(probs, axis=1)

            high_prob_masks = (probs > 0.9).astype(np.uint8)
            preds           = preds * high_prob_masks
            for i, pred in enumerate(preds):
                no_value_mask   = dataset.get_mask(batch_idx * args.batch_size + i)
                pred            = pred * no_value_mask
                label           = Image.fromarray(pred)
                
                basename        = dataset.get_basename(batch_idx * args.batch_size + i)
                label.save(os.path.join(args.save_dir, "%s.png" % basename))
            
if __name__ == "__main__":
    if args.train:
        train()
    elif args.predict:
        predict()
    else:
        print("Please chose --train, --predict. Mutualy exclusive!")