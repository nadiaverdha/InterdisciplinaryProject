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
from csv import DictWriter
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
from torch.nn import functional as F

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A

from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt


def dice_coeff(input, target):
    inter = 2 * (input * target).sum()
    sets_sum = input.sum() + target.sum()
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + 1) / (sets_sum + 1)
    return dice.item()


def dice_loss(input, target):
    fn = dice_coeff
    return 1 - fn(input, target)


def iou(logits, targets, smooth=1):
    intersection = (logits * targets).sum()
    union = logits.sum() + targets.sum() - intersection + smooth
    if union == 0:
        return 1
    iou = intersection / union
    return iou.item()


def pixel_accuracy(logits, targets):
    preds_flat = logits.view(-1)
    targets_flat = targets.view(-1)

    correct = torch.sum(preds_flat == targets_flat)
    total_pixels = targets_flat.numel()
    accuracy = correct.float() / total_pixels

    return accuracy.item()


def train_evaluate(model, epochs, trainloader, valloader, optimizer, criterion, grad_scaler, scheduler, dict_file,
                   model_file, best_dice=0, patience=10):
    train_losses, val_losses = [], []
    train_dices, train_l_dices, val_dices, val_l_dices = [], [], [], []
    train_ious, train_l_ious, val_ious, val_l_ious = [], [], [], []
    train_accs, train_l_accs, val_accs, val_l_accs = [], [], [], []
    for epoch in tqdm(range(epochs)):
        ###################
        # train the model #
        ###################
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
            logits = model(images)
            loss = criterion(logits, masks.float())
            loss += dice_loss(logits, masks.float())

            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            train_loss += loss.item()
            train_dice += dice_coeff((logits > 0.5), masks)
            train_l_dice += dice_coeff((logits > 0.5), lacken_masks)
            train_iou += iou((logits > 0.5), masks)
            train_l_iou += iou((logits > 0.5), lacken_masks)
            train_acc += pixel_accuracy((logits > 0.5), masks)
            train_l_acc += pixel_accuracy((logits > 0.5), lacken_masks)

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

        ######################
        # validate the model #
        ######################
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
                loss = criterion(logits, masks.float())
                loss += dice_loss(logits, masks.float())

                val_loss += loss.item()
                val_dice += dice_coeff((logits > 0.5), masks)
                val_l_dice += dice_coeff((logits > 0.5), lacken_masks)
                val_iou += iou((logits > 0.5), masks)
                val_l_iou += iou((logits > 0.5), lacken_masks)
                val_acc += pixel_accuracy((logits > 0.5), masks)
                val_l_acc += pixel_accuracy((logits > 0.5), lacken_masks)
        scheduler.step(val_dice)
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

        print(f" \n Epoch: {epoch + 1} ")
        print(
            f"TRAIN FUll: Train Loss: {train_loss:.4f} | Train DICE Coeff: {train_dice:.4f}  | Train IoU Coeff: {train_iou:.4f}|  | Train Accuracy: {train_acc * 100:.2f} ")
        print(
            f"TRAIN LACKENS: Train DICE Coeff: {train_l_dice:.4f}  | Train IoU Coeff: {train_l_iou:.4f}|  | Train Accuracy: {train_l_acc * 100:.2f} ")
        print(
            f"VAL FULL: Val Loss: {val_loss:.4f} | Val DICE Coeff: {val_dice:.4f} | Val IoU Coeff: {val_iou:.4f}| Val Accuracy: {val_acc * 100:.2f}| ")
        print(
            f"VAL LACKENS: Val DICE Coeff: {val_l_dice:.4f} | Val IoU Coeff: {val_l_iou:.4f}| Val Accuracy: {val_l_acc * 100:.2f}| ")

        d = {'epoch': epoch, 'train loss': train_loss, 'valid loss': val_loss, 'train_dice': train_dice,
             'train_l_dice': train_l_dice, 'val_dice': val_dice, 'val_l_dice': val_l_dice}

        with open(dict_file, 'a') as f:
            dictwriter_object = DictWriter(f,
                                           fieldnames=['epoch', 'train loss', 'valid loss', 'train_dice',
                                                       'train_l_dice', 'val_dice',
                                                       'val_l_dice'])
            dictwriter_object.writerow(d)

        if val_dice > best_dice:
            best_dice = val_dice
            patience = patience
        else:
            patience -= 1

        if patience <= 0:
            print("Early Stopping")
            break

        torch.save(model.state_dict(), model_file)
