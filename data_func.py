import os
import glob
from osgeo import gdal
import osgeo.gdal
import pprint
import numpy as np
from yeoda.datacube import DataCubeReader
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
import pandas as pd
import time
import rasterio
import glob
import cv2
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
import fiona
from fiona.transform import transform_geom


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
        if (_.month == 4) | (_.month == 9):
            merge('test', type, row, notmask)
        elif (_.month == 5) | (_.month == 10):
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
        os.makedirs(directory, exist_ok=True)
        output_path = directory + '/' + "VH_mask" + '_' + str(row.name) + '_merged.tif'
        command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]}"
        os.system(command)
        output_path = directory + '/' + "VV_mask" + '_' + str(row.name) + '_merged.tif'
        command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]}"
        os.system(command)


def mask_files(shapefile, input_dir, output_dir):
    tif_files = os.listdir(input_dir)
    for tiff in tif_files:
        output_filepath = os.path.join(output_dir, tiff)
        if tiff.endswith(('.tif', '.tiff')):
            # print(output_filepath)
            with rasterio.open(output_filepath) as src:
                raster_crs = src.crs
            with fiona.open('shape_file.shp', "r") as shapefile:
                shapes = []
                for feature in shapefile:
                    geom = feature["geometry"]
                    if shapefile.crs != raster_crs:
                        geom = transform_geom(shapefile.crs, raster_crs, geom)
                    shapes.append(geom)

            with rasterio.open(output_filepath) as src:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                out_meta = src.meta.copy()

            # Update the metadata and save the result
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            with rasterio.open(output_filepath, "w", **out_meta) as dest:
                dest.write(out_image)

def create_lacken_mask(input_dir, output_dir):
    tif_files = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for tif_file in tif_files:
        with rasterio.open((input_dir + tif_file)) as src:
            water_mask = src.read(1)
        water_mask_uint8 = np.uint8(water_mask)
        contours, _ = cv2.findContours(water_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lacken_mask = np.zeros_like(water_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000000:
                cv2.drawContours(lacken_mask, [contour], -1, 1, thickness=-1)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": lacken_mask.shape[0], "width": lacken_mask.shape[1], "count": 1})

        output_filename = os.path.splitext(os.path.basename(tif_file))[0] + "_lacken_mask.tif"
        output_filepath = os.path.join(output_dir, output_filename)

        with rasterio.open(output_filepath, "w", **out_meta) as dst:
            dst.write(lacken_mask.astype(rasterio.uint8), 1)


def make_list(dir, mask_dir, name):
    img_list = sorted(glob.glob(dir))
    mask_list = sorted(glob.glob(mask_dir))
    file_num = len(img_list)
    output_filename = "./" + name + "_images.txt"
    with open(output_filename, 'w') as f:
        for s in range(file_num):
            f.write(img_list[s] + '\t' + mask_list[s] + '\n')


class ImageDataset(Dataset):
    def __init__(self, images_folder, mask_folder, lacken_mask_folder, transform=None):
        self.images = os.listdir(images_folder)
        self.images_folder = images_folder
        self.masks = os.listdir(mask_folder)
        self.transform = transform
        self.mask_folder = mask_folder
        self.lacken_mask_folder = lacken_mask_folder
        self.lacken_masks = os.listdir(self.lacken_mask_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.images[idx].endswith(('.tif', '.tiff')):
            image_path = os.path.join(self.images_folder, self.images[idx])
            mask_path = os.path.join(self.mask_folder, self.masks[idx])
            lacken_mask_path = os.path.join(self.lacken_mask_folder, self.lacken_masks[idx])

            image = rasterio.open(image_path).read(1)
            mask = rasterio.open(mask_path).read(1)
            lacken_mask = rasterio.open(lacken_mask_path).read(1)
            image = np.array(image, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)
            lacken_mask = np.array(lacken_mask, dtype=np.float32)

            if self.transform:
                augmented = self.transform(image=image, masks=[mask, lacken_mask])
                image = augmented['image']
                mask = augmented['masks'][0]
                lacken_mask = augmented['masks'][1]

            return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), torch.from_numpy(
                lacken_mask).unsqueeze(0)


def visualize_augmentations(dataset, idx=0, samples=3):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=3, figsize=(10, 13))
    for i in range(samples):
        # print(i)
        image, mask, lacken_mask = dataset[idx]
        # print(image.squeeze(0).shape)
        ax[i, 0].imshow(image.squeeze(0))
        ax[i, 1].imshow(mask.squeeze(0), interpolation="nearest")
        ax[i, 2].imshow(lacken_mask.squeeze(0), interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("mask")
        ax[i, 2].set_title("Lacken mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()


def visualize_offline_augmentations(dataset, samples=3):
    dataset = copy.deepcopy(dataset)
    figure, ax = plt.subplots(nrows=samples, ncols=3, figsize=(10, 13))
    for i in range(3):
        image, mask, lacken_mask = dataset[i]
        # print(image.squeeze(0).shape)
        ax[i, 0].imshow(image.squeeze(0))
        ax[i, 1].imshow(mask.squeeze(0), interpolation="nearest")
        ax[i, 2].imshow(lacken_mask.squeeze(0), interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("mask")
        ax[i, 2].set_title("Lacken mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()

