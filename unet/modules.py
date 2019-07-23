import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from dataset import *
import os
import csv

from tqdm import tqdm



def train_model(model, train_loader, criterion, optimizer, scheduler, keep_rate=1.0):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        train_loader (DataLoader): training dataset
    """    
    model.train()
    epoch_loss = []
    pbar = tqdm(train_loader)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)

        # calculate loss and remove loss of ignore_index
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()

        epoch_loss.append(loss.item())
    scheduler.step(np.mean(epoch_loss))

def get_loss(model, data, criterion):
    """
        Calculate loss over train set
    """
    model.eval()
    total_loss = 0
    for batch, (images, labels) in enumerate(data):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model(images)

            loss = criterion(outputs, labels)
                        
            total_loss += loss.item()
    return total_loss / (batch + 1)

def test_model(model_path, data_loader, epoch, save_folder_name='prediction'):
    model = torch.load(model_path)
    model = model.cuda()

    model.eval()
    for batch, (images_t) in enumerate(data_loader):
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_t.size()[1]):
            with torch.no_grad():
                image_t = Variable(images_t[:, index, :, :].unsqueeze(0).cuda())
                # print(image_v.shape, mask_v.shape)
                output_t = model(image_t)
                output_t = torch.argmax(output_t, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_t))
        im_name = batch  # TODO: Change this to real image name so we know
        _ = save_prediction_image(stacked_img, im_name, epoch, save_folder_name)
    print("Finish Prediction!")

def save_prediction_image(stacked_img, im_name, epoch, save_folder_name="result_images", save_im=True):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
    """
    div_arr = division_array(388, 2, 2, 512, 512)
    img_cont = image_concatenate(stacked_img.cpu().data.numpy(), 2, 2, 512, 512)
    img_cont = polarize((img_cont)/div_arr)*255
    img_cont_np = img_cont.astype('uint8')
    img_cont = Image.fromarray(img_cont_np)
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    img_cont.save(desired_path + export_name)
    return img_cont_np

def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img
