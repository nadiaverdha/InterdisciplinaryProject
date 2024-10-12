import torch
import copy
from data_func import merge_files_and_split, ImageDataset, make_list, visualize_augmentations
from model import DownSample, DoubleConv, OutConv, UpSample, UNet
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch import optim, nn
import cv2
import time
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.optim import Adam
from torch import permute
from torch import nan_to_num
from torch.nn import BCEWithLogitsLoss
from torchvision.transforms import ToTensor
import rasterio
from torch import nn
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
from torch.nn import functional as F

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A

from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = 2 * (logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union

        loss = nn.BCELoss()
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss


def dice_coeff(logits, targets):
    logits = (logits > 0.5).float()
    intersection = 2 * (logits * targets).sum()
    union = (logits + targets).sum()
    if union == 0:
        return 1
    dice_coeff = intersection / union
    return dice_coeff.item()


def iou(logits, targets, smooth=1):
    logits = (logits > 0.5).float()
    intersection = 2 * (logits * targets).sum()
    union = (logits + targets).sum() - intersection
    if union == 0:
        return 1
    iou = intersection / union

    return iou.item()


def pixel_accuracy(logits, targets):
    preds = (logits > 0.5).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    correct = torch.sum(preds_flat == targets_flat)
    total_pixels = targets_flat.numel()
    accuracy = correct.float() / total_pixels

    return accuracy.item()


def train(model, trainloader, valloader, optimizer, loss, epochs=10):
    train_losses, val_losses = [], []
    train_dices, train_l_dices, val_dices, val_l_dices = [], [], [], []
    train_ious, train_l_ious, val_ious, val_l_ious = [], [], [], []
    train_accs, train_l_accs, val_accs, val_l_accs = [], [], [], []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_acc = 0
        train_l_acc = 0
        train_l_dice = 0
        train_l_iou = 0

        for i, (images, masks, lacken_masks) in enumerate(trainloader):
            images, masks, lacken_masks = images.to(device), masks.to(device), lacken_masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, masks)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_dice += dice_coeff(logits, masks)
            train_iou += iou(logits, masks)
            train_acc += pixel_accuracy(logits, masks)
            train_l_acc += pixel_accuracy(logits, lacken_masks)
            train_l_dice += dice_coeff(logits, lacken_masks)
            train_l_iou += iou(logits, lacken_masks)
        train_loss /= len(trainloader)
        train_dice /= len(trainloader)
        train_iou /= len(trainloader)
        train_acc /= len(trainloader)
        train_l_acc /= len(trainloader)
        train_l_dice /= len(trainloader)
        train_l_iou /= len(trainloader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        train_ious.append(train_iou)
        train_accs.append(train_acc)
        train_l_accs.append(train_l_acc)
        train_l_ious.append(train_l_iou)
        train_l_dices.append(train_l_dice)

        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        val_acc = 0
        val_l_dice = 0
        val_l_iou = 0
        val_l_acc = 0
        with torch.no_grad():
            for i, (images, masks, lacken_masks) in enumerate(valloader):
                images, masks, lacken_masks = images.to(device), masks.to(device), lacken_masks.to(device)
                logits = model(images)
                l = loss(logits, masks)
                val_loss += l.item()
                val_dice += dice_coeff(logits, masks)
                val_iou += iou(logits, masks)
                val_acc += pixel_accuracy(logits, masks)
                val_l_dice += dice_coeff(logits, lacken_masks)
                val_l_iou += iou(logits, lacken_masks)
                val_l_acc += pixel_accuracy(logits, lacken_masks)
        val_loss /= len(valloader)
        val_dice /= len(valloader)
        val_iou /= len(valloader)
        val_acc /= len(valloader)
        val_l_acc /= len(valloader)
        val_l_dice /= len(valloader)
        val_l_iou /= len(valloader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        val_accs.append(val_acc)
        val_l_accs.append(val_l_acc)
        val_l_dices.append(val_l_dice)
        val_l_ious.append(val_l_iou)
        print(f"Epoch: {epoch + 1} ")
        print(
            f"TRAIN FUll: Train Loss: {train_loss:.4f} | Train DICE Coeff: {train_dice:.4f}  | Train IoU Coeff: {train_iou:.4f}|  | Train Accuracy: {train_acc * 100:.2f} ")
        print(
            f"TRAIN LACKENS: Train DICE Coeff: {train_l_dice:.4f}  | Train IoU Coeff: {train_l_iou:.4f}|  | Train Accuracy: {train_l_acc * 100:.2f} ")
        print(
            f"VAL FULL: Val Loss: {val_loss:.4f} | Val DICE Coeff: {val_dice:.4f} | Val IoU Coeff: {val_iou:.4f}| Val Accuracy: {val_acc * 100:.2f}| ")
        print(
            f"VAL LACKENS: Val DICE Coeff: {val_l_dice:.4f} | Val IoU Coeff: {val_l_iou:.4f}| Val Accuracy: {val_l_acc * 100:.2f}| ")

    return train_losses, train_dices, val_losses, val_dices, train_l_dices, train_l_ious, train_l_accs, val_l_dice, val_l_ious, val_l_accs
