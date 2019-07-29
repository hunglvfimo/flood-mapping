import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_bce_loss(outputs, labels):
    # mask for labeled pixel
    mask    = torch.max(labels, dim=1)[0]

    loss    = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
    loss    = torch.sum(loss, dim=1)
    
    # mask-out unlabel pixels
    loss    = loss[mask > 0]
    loss    = loss.mean()
    return loss

def masked_dice_loss(outputs, labels):
    smooth = 1

    # mask for labeled pixel
    mask            = torch.max(labels, dim=1)[0] # Batch_size x Height x Width

    # Intersction
    intersection    = outputs * labels # Batch_size x C x Height x Width
    intersection    = torch.sum(intersection, dim=1) # Batch_size x Height x Width

    # Union
    u_outputs       = torch.sum(outputs, dim=1)
    u_labels        = torch.sum(labels, dim=1)

    # mask-out unlabel pixels
    intersection    = intersection[mask > 0]
    u_outputs       = u_outputs[mask > 0]
    u_labels        = u_labels[mask > 0]

    return 1 - (2. * intersection.sum() + smooth) / (u_outputs.sum() + u_labels.sum() + smooth)

def masked_dbce_loss(outputs, labels, bce_weigth=0.5):
    """Dice loss + BCE loss"""
    # mask for labeled pixel
    # mask        = torch.max(labels, dim=1)[0] # Batch_size x Height x Width

    bce_loss    = masked_bce_loss(outputs, labels)
    dice_loss   = masked_dice_loss(outputs, labels)

    return bce_weigth * bce_loss + (1 - bce_weigth) * dice_loss

