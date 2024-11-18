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
from sklearn.metrics import balanced_accuracy_score

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
import numpy as np
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt


# def dice_coeff(input, target):
#     inter = 2 * (input * target).sum()
#     sets_sum = input.sum() + target.sum()
#     if sets_sum == 0:
#         return 1
#     dice = (inter + 1) / (sets_sum + 1)
#     return dice.item()

def dice_coeff(input, target, smooth=1):
    # if input.dim() == 2:
    sum_dim = (-1, -2)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + 1) / (sets_sum + 1)
    return dice.mean()


def dice_loss(input, target):
    fn = dice_coeff
    return 1 - fn(input, target)


def iou(logits, targets, smooth=1):
    intersection = (logits * targets).sum()
    union = logits.sum() + targets.sum() - intersection
    if union == 0:
        return 1
    iou = (intersection + 1) / (union + 1)
    return iou.item()


def pixel_accuracy(logits, targets):
    preds_flat = logits.view(-1)
    targets_flat = targets.view(-1)
    correct = torch.sum(preds_flat == targets_flat)
    total_pixels = targets_flat.numel()
    accuracy = correct.float() / total_pixels

    return accuracy.item()


def balanced_accuracy(logits, targets):
    preds_flat = logits.view(-1)
    targets_flat = targets.view(-1)

    TP = torch.sum((preds_flat == 1) & (targets_flat == 1)).float()
    TN = torch.sum((preds_flat == 0) & (targets_flat == 0)).float()
    FP = torch.sum((preds_flat == 1) & (targets_flat == 0)).float()
    FN = torch.sum((preds_flat == 0) & (targets_flat == 1)).float()

    P = TP + FN
    N = TN + FP

    if P == 0 or N == 0:
        return float('nan')

    sensitivity = TP / P
    specificity = TN / N

    balanced_acc = (sensitivity + specificity) / 2.0

    return balanced_acc.item()
    # preds_flat = (logits.view(-1) > 0.5).int()  # or logits.argmax(dim=1) for multiclass
    # targets_flat = targets.view(-1)
    # return balanced_accuracy_score(preds_flat, targets_flat)


grad_scaler = torch.cuda.amp.GradScaler()


def train_evaluate(model, epochs, trainloader, valloader, optimizer, criterion, dict_file,
                   model_file, best_dice=0):
    train_losses, val_losses = [], []
    train_dices, train_l_dices, val_dices, val_l_dices = [], [], [], []
    train_ious, train_l_ious, val_ious, val_l_ious = [], [], [], []
    train_accs, train_l_accs, val_accs, val_l_accs = [], [], [], []
    train_bal_accs, val_bal_accs, train_l_bal_accs, val_l_bal_accs = [], [], [], []
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
        train_bal_acc = 0
        train_l_bal_acc = 0

        for i, (images, masks, lacken_masks) in enumerate(trainloader):
            images, masks, lacken_masks = images.to(device), masks.to(device), lacken_masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks.float())
            loss += dice_loss(F.sigmoid(logits), masks.float())

            optimizer.zero_grad()

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            mask_pred = (F.sigmoid(logits) > 0.5).float()

            train_loss += loss.item()
            train_dice += dice_coeff(mask_pred, masks)
            train_l_dice += dice_coeff(mask_pred, lacken_masks)
            train_iou += iou(mask_pred, masks)
            train_l_iou += iou(mask_pred, lacken_masks)
            train_acc += pixel_accuracy(mask_pred, masks)
            train_l_acc += pixel_accuracy(mask_pred, lacken_masks)
            train_bal_acc += balanced_accuracy(mask_pred, masks)
            train_l_bal_acc += balanced_accuracy(mask_pred, lacken_masks)
        train_loss /= len(trainloader)
        train_dice /= len(trainloader)
        train_iou /= len(trainloader)
        train_acc /= len(trainloader)
        train_l_acc /= len(trainloader)
        train_l_dice /= len(trainloader)
        train_l_iou /= len(trainloader)
        train_l_bal_acc /= len(trainloader)
        train_bal_acc /= len(trainloader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        train_ious.append(train_iou)
        train_accs.append(train_acc)
        train_l_accs.append(train_l_acc)
        train_l_ious.append(train_l_iou)
        train_l_dices.append(train_l_dice)
        train_bal_accs.append(train_bal_acc)
        train_l_bal_accs.append(train_l_bal_acc)

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
        val_l_bal_acc = 0
        val_bal_acc = 0
        with torch.no_grad():
            for i, (images, masks, lacken_masks) in enumerate(valloader):
                images, masks, lacken_masks = images.to(device), masks.to(device), lacken_masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks.float())
                mask_pred = (F.sigmoid(logits) > 0.5).float()
                loss += dice_loss(F.sigmoid(logits), masks.float())
                val_loss += loss.item()
                val_dice += dice_coeff(mask_pred, masks)
                val_l_dice += dice_coeff(mask_pred, lacken_masks)
                val_iou += iou(mask_pred, masks)
                val_l_iou += iou(mask_pred, lacken_masks)
                val_acc += pixel_accuracy(mask_pred, masks)
                val_l_acc += pixel_accuracy(mask_pred, lacken_masks)
                val_l_bal_acc += balanced_accuracy(mask_pred, masks)
                val_bal_acc += balanced_accuracy(mask_pred, lacken_masks)
        val_loss /= len(valloader)
        val_dice /= len(valloader)
        val_iou /= len(valloader)
        val_acc /= len(valloader)
        val_l_acc /= len(valloader)
        val_l_dice /= len(valloader)
        val_l_iou /= len(valloader)
        val_l_bal_acc /= len(valloader)
        val_bal_acc /= len(valloader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        val_accs.append(val_acc)
        val_l_accs.append(val_l_acc)
        val_l_dices.append(val_l_dice)
        val_l_ious.append(val_l_iou)
        val_l_bal_accs.append(val_l_bal_acc)
        val_bal_accs.append(val_bal_acc)

        print(f" \n Epoch: {epoch + 1} ")
        print(
            f"TRAIN FULL: Train Loss: {train_loss:.4f} | Train DICE Coeff: {train_dice:.4f}  | Train IoU Coeff: {train_iou:.4f}| | Train Bal Accuracy: {train_bal_acc * 100:.2f} | Train Accuracy: {train_acc * 100:.2f} ")
        print(
            f"TRAIN LACKENS: Train DICE Coeff: {train_l_dice:.4f}  | Train IoU Coeff: {train_l_iou:.4f} | Train Bal Accuracy: {train_l_bal_acc * 100:.2f} |  Train Accuracy: {train_l_acc * 100:.2f} ")
        print(
            f"VAL FULL: Val Loss: {val_loss:.4f} | Val DICE Coeff: {val_dice:.4f} | Val IoU Coeff: {val_iou:.4f} | Val Bal Accuracy: {val_bal_acc * 100:.2f} | Val Accuracy: {val_acc * 100:.2f} ")
        print(
            f"VAL LACKENS: Val DICE Coeff: {val_l_dice:.4f} | Val IoU Coeff: {val_l_iou:.4f} | Val Bal Accuracy: {val_l_bal_acc * 100:.2f} |  Val Accuracy: {val_l_acc * 100:.2f}| ")

        d = {'epoch': epoch, 'train loss': train_loss, 'valid loss': val_loss, 'train_dice': train_dice,
             'train_l_dice': train_l_dice, 'train_bal_acc': train_bal_acc, 'train_l_bal_acc': train_l_bal_acc,
             'val_dice': val_dice, 'val_l_dice': val_l_dice, 'val_l_bal_acc': val_l_bal_acc, 'val_bal_acc': val_bal_acc}

        with open(dict_file, 'a') as f:
            dictwriter_object = DictWriter(f,
                                           fieldnames=['epoch', 'train loss', 'valid loss', 'train_dice',
                                                       'train_l_dice', 'train_bal_acc', 'train_l_bal_acc',
                                                       'val_dice', 'val_l_dice', 'val_l_bal_acc', 'val_bal_acc'])
            dictwriter_object.writerow(d)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), model_file)


def display_batch(images, masks, pred, lacken_masks):
    print(f"Dice Coefficient: {dice_coeff(pred.to(device), masks.to(device))}")
    print(f"Dice Coefficient Lackens: {dice_coeff(pred.to(device), lacken_masks.to(device))}")
    print(f"IoU: {iou(pred.to(device), masks.to(device))}")
    print(f"IoU Lackens: {iou(pred.to(device), lacken_masks.to(device))}")
    print(f"Accuracy: {balanced_accuracy(pred.to(device), masks.to(device))}")
    print(f"Accuracy Lackens: {balanced_accuracy(pred.to(device), lacken_masks.to(device))}")
    images = images.permute(0, 2, 3, 1)
    masks = masks.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    pred = pred.cpu().numpy()

    images = np.concatenate(images, axis=1)
    masks = np.concatenate(masks, axis=1)
    pred = np.concatenate(pred, axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    fig.tight_layout()
    ax[0].imshow(images)
    ax[0].set_title('Images')
    ax[1].imshow(masks, cmap='gray')
    ax[1].set_title('Masks')
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Predictions')

    plt.show()


def display_batch_all(images, masks, pred, pred2, pred3, pred4, lacken_masks, i):
    images = images.permute(0, 2, 3, 1)
    masks = masks.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)
    pred4 = pred4.permute(0, 2, 3, 1)
    pred2 = pred2.permute(0, 2, 3, 1)
    pred3 = pred3.permute(0, 2, 3, 1)

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    pred = pred.cpu().numpy()
    pred2 = pred2.cpu().numpy()
    pred3 = pred3.cpu().numpy()
    pred4 = pred4.cpu().numpy()

    images = np.concatenate(images, axis=1)
    masks = np.concatenate(masks, axis=1)
    pred = np.concatenate(pred, axis=1)
    pred2 = np.concatenate(pred2, axis=1)
    pred3 = np.concatenate(pred3, axis=1)
    pred4 = np.concatenate(pred4, axis=1)

    fig, ax = plt.subplots(6, 1, figsize=(30, 8))
    fig.tight_layout()
    ax[0].imshow(images)
    ax[0].set_title('Images')
    ax[1].imshow(masks, cmap='gray')
    ax[1].set_title('Masks')
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Predictions Model I' )
    ax[3].imshow(pred2, cmap='gray')
    ax[3].set_title('Predictions Model II' )
    ax[4].imshow(pred3, cmap='gray')
    ax[4].set_title('Predictions Model III' )
    ax[5].imshow(pred4, cmap='gray')
    ax[5].set_title('Predictions Model IV' )
    plt.savefig('./inf' + '.png')
    plt.show()