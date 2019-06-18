import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable

from advanced_model import UNet
# from loss import CrossEntropyLoss
from dataset import *
from modules import *
from save_history import *

data_dir = os.path.join("..", "..", "..", "data", "processed", "PI_SAR2_FINE")
history_dir = os.path.join("..", "history")

if __name__ == "__main__":
    # Dataset begin
    train_dataset = SEMDataset(os.path.join(data_dir, "train", "img"), os.path.join(data_dir, "train", "mask"), is_train=True)
    val_dataset = SEMDataset(os.path.join(data_dir, "val", "img"), os.path.join(data_dir, "val", "mask"), is_train=False)
    # Dataset end

    # Dataloader begins
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=0, batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=0, batch_size=2, shuffle=False)
    # Dataloader end

    # Model
    model = UNet(n_class=3)
    model = model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Optimizerd
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    # Parameters
    epoch_start = 0
    epoch_end = 100

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_dir = os.path.join(history_dir, "RMS")

    save_file_name = os.path.join(save_dir, "history.csv")

    # Saving images and models directories
    model_save_dir = os.path.join(save_dir, "models")
    image_save_path = os.path.join(save_dir, "result_images3")

    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # train the model
        train_model(model, train_loader, criterion, optimizer)
        # Validation every 5 epoch
        if (i + 1) % 5 == 0:
            train_acc, train_loss = get_loss_train(model, train_loader, criterion)
            val_acc, val_loss = validate_model(model, val_loader, criterion, i + 1, True, image_save_path)
            
            print('Epoch %d, Train loss: %.3f, Train acc: %.3f, Val loss: %.3f, Val acc: %.3f' % (i + 1, train_loss, train_acc, val_loss, val_acc))

            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i + 1) % 10 == 0:  # save model every 10 epoch
                save_models(model, model_save_dir, i + 1)