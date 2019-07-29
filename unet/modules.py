import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from tqdm import tqdm

def train_model(model, data_loader, criterion, optimizer, scheduler):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        train_loader (DataLoader): training dataset
    """    
    model.train()
    epoch_loss = []
    
    pbar = tqdm(data_loader)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)

        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("%.3f" % loss.item())
        epoch_loss.append(loss.item())
        
    scheduler.step(np.mean(epoch_loss))
    return np.mean(epoch_loss)

def evaluate_model(model, data_loader, criterion, metric=False):
    """
        Calculate loss over train set
    """
    total_loss  = []
    total_acc   = []
    total_pixel = []

    model.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(data_loader):
            images      = images.cuda()
            labels      = labels.cuda()
            
            outputs     = model(images)
            if criterion is not None:
                loss    = criterion(outputs, labels)
                total_loss.append(loss.item())
            else:
                total_loss.append(0)

            outputs     = outputs.cpu().numpy()
            predicted   = np.argmax(outputs, axis=1)

            labels      = labels.cpu().numpy()
            
            # mask out un-labledl pixels
            mask        = np.max(labels, axis=1)
            labels      = np.argmax(labels, axis=1)
            
            num_pixel   = np.sum(mask)
            total_pixel.append(num_pixel)

            # calculate acc
            matches     = (predicted == labels).astype(np.uint8)
            matches     = matches * mask

            num_correct = np.sum(matches)
            total_acc.append(num_correct / num_pixel)
            
    total_pixel = total_pixel / np.sum(total_pixel)
    return np.sum(total_loss * total_pixel), np.sum(total_acc * total_pixel)
