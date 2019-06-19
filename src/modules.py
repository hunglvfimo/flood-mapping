import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import *
import os
import csv
from tqdm import tqdm

import torch.nn as nn

from accuracy import accuracy_check, accuracy_check_for_batch

def train_model(model, train_loader, criterion, optimizer, scheduler, ignore_index=0, keep_rate=1.0):
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
        # loss[labels == ignore_index] = 0
        # loss = loss.view(-1)
        # loss = loss[loss != 0]
        # # select samples with higher confidence (lower loss) for training
        # if keep_rate < 1.0:
        #     loss_ind_sorted = np.argsort(loss.cpu().data.numpy())
        #     loss_ind_sorted = torch.LongTensor(loss_ind_sorted.copy()).cuda()
        #     num_keep = int(keep_rate * len(loss))

        #     loss = loss[loss_ind_sorted[:num_keep]]

        # loss = loss.mean()
        pbar.set_description("[kr=%.2f]%.3f" % (keep_rate, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()

        epoch_loss.append(loss.item())
    scheduler.step(np.mean(epoch_loss))

def get_loss(model, data_train, criterion):
    """
        Calculate loss over train set
    """
    model.eval()
    total_loss = 0
    for batch, (images, masks) in enumerate(data_train):
        with torch.no_grad():
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss = loss.mean()
                        
            total_loss = total_loss + loss.cpu().item()
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
