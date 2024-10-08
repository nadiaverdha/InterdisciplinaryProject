import os
import glob
from osgeo import gdal
import pprint
import numpy as np
from yeoda.datacube import DataCubeReader
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
import pandas as pd
import time
import rasterio
import glob
from rasterio.plot import show
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.plot import show
import tifftools
from torch.utils.data import TensorDataset, DataLoader, Dataset
import tensorflow as tf
import torch
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# to merge the files
def merge_files(df, type='VV'):
    output_directory = './merged_tiffs'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    grouped = df[df['filepath'].str.endswith('.tif')]
    # grouped['date'] = grouped['time'].dt.date
    grouped.loc[:, 'date'] = grouped['time'].dt.date
    grouped_type = grouped[grouped['filepath'].str.contains(type)]
    grouped_type = grouped_type.groupby('date').agg({'filepath': list})
    for _, row in grouped_type.iterrows():
        output_path = output_directory + '/' + type + '_' + str(row.name) + '_merged.tiff'
        command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]} {row.filepath[2]} {row.filepath[3]}"
        os.system(command)


# to merge the files and split into train, validation and test
def merge_files_and_split(df, type='VV', notmask=True):
    grouped = df[df['filepath'].str.endswith('.tif')]
    grouped.loc[:, 'date'] = grouped['time'].dt.date
    if notmask:
        grouped = grouped[grouped['filepath'].str.contains(type)]
    grouped = grouped.groupby('date').agg({'filepath': list})
    for _, row in grouped.iterrows():
        if (_.month == 4) | (_.month == 7):
            merge('test', type, row, notmask)
        elif (_.month == 5) | (_.month == 8):
            merge('val', type, row, notmask)
        else:
            merge('train', type, row, notmask)


def merge(directory, type, row, notmask):
    if notmask:
        os.makedirs(directory, exist_ok=True)
        output_path = directory + '/' + type + '_' + str(row.name) + '_merged.tiff'
        command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]} {row.filepath[2]} {row.filepath[3]}"
        os.system(command)
    else:
        directory = directory + "_mask"
        os.makedirs( directory, exist_ok=True)
        output_path = directory + '/' + "VH_mask" + '_' + str(row.name) + '_merged.tiff'
        command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]}"
        os.system(command)
        output_path = directory + '/' + "VV_mask" + '_' + str(row.name) + '_merged.tiff'
        command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]}"
        os.system(command)

def make_list(dir, mask_dir, name):
    img_list = sorted(glob.glob(dir))
    mask_list = sorted(glob.glob(mask_dir))
    file_num = len(img_list)
    output_filename = "./" + name + "_images.txt"
    with open(output_filename, 'w') as f:
        for s in range(file_num):
            f.write(img_list[s] + '\t' + mask_list[s] + '\n')


class ImageDataset(Dataset):
    def __init__(self, images_folder, mask_folder, transform=None):
        self.images = os.listdir(images_folder)
        self.images_folder = images_folder
        self.masks = os.listdir(mask_folder)
        self.transform = transform
        self.mask_folder = mask_folder

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.images[idx])
        mask_path = os.path.join(self.mask_folder, self.masks[idx])
        image = rasterio.open(image_path).read(1)
        mask = rasterio.open(mask_path).read(1)
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return torch.from_numpy(image), torch.from_numpy(mask)

def visualize_augmentations(dataset, idx=0, samples=3):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(5, 6))
    for i in range(samples):
        # print(i)
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()
