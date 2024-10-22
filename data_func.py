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
from PIL import Image, ImageOps
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
import scipy.ndimage as ndi

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
        if self.images[idx].endswith(('.tif', '.tiff')) and self.masks[idx].endswith(('.tif', '.tiff')) and self.lacken_masks[idx].endswith(('.tif', '.tiff')):
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


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform(x, rotation_deg, fill_mode='nearest', cval=0):
    channel_axis = 0
    row_axis = 1
    col_axis = 2

    theta = np.deg2rad(rotation_deg)
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    return apply_transform(x, transform_matrix, channel_axis, fill_mode=fill_mode, cval=cval)


def augment_images_and_masks(images_folder, mask_folder, lacken_mask_folder, save_folder,
                             rotation_deg=[-15, -10, -5, 0, 5, 10, 15], X_flip=[0, 1], Y_flip=[0, 1]):
    os.makedirs(os.path.join(save_folder, images_folder), exist_ok=True)
    os.makedirs(os.path.join(save_folder, mask_folder), exist_ok=True)
    os.makedirs(os.path.join(save_folder, lacken_mask_folder), exist_ok=True)

    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.tif', '.tiff'))]
    masks = [f for f in os.listdir(mask_folder) if f.endswith(('.tif', '.tiff'))]
    lacken_masks = [f for f in os.listdir(lacken_mask_folder) if f.endswith(('.tif', '.tiff'))]
    for i in range(len(image_files)):
        print(f'Processing image and mask num: {i}')
        for r in range(len(rotation_deg)):
            for x in range(len(X_flip)):
                for y in range(len(Y_flip)):
                    # Load image, mask, and lacken mask
                    image_path = os.path.join(images_folder, image_files[i])
                    mask_path = os.path.join(mask_folder, masks[i])
                    lacken_mask_path = os.path.join(lacken_mask_folder, lacken_masks[i])

                    image = Image.open(image_path)
                    mask = Image.open(mask_path)
                    lacken_mask = Image.open(lacken_mask_path)

                    # if X_flip[x] == 1:
                    #     image = ImageOps.mirror(image)
                    #     mask = ImageOps.mirror(mask)
                    #     lacken_mask = ImageOps.mirror(lacken_mask)
                    # if Y_flip[y] == 1:
                    #     image = ImageOps.flip(image)
                    #     mask = ImageOps.flip(mask)
                    #     lacken_mask = ImageOps.flip(lacken_mask)

                    # Convert to numpy array
                    image = np.array(image, dtype=np.float32)
                    mask = np.array(mask, dtype=np.float32)
                    lacken_mask = np.array(lacken_mask, dtype=np.float32)

                    # If single-channel, expand dimensions
                    if len(image.shape) == 2:
                        image = np.expand_dims(image, axis=2)
                    if len(mask.shape) == 2:
                        mask = np.expand_dims(mask, axis=2)
                    if len(lacken_mask.shape) == 2:
                        lacken_mask = np.expand_dims(lacken_mask, axis=2)

                    # Apply transformation
                    image = np.moveaxis(image, 2, 0)
                    mask = np.moveaxis(mask, 2, 0)
                    lacken_mask = np.moveaxis(lacken_mask, 2, 0)

                    image_transformed = transform(image, rotation_deg=rotation_deg[r])
                    mask_transformed = transform(mask, rotation_deg=rotation_deg[r])
                    lacken_mask_transformed = transform(lacken_mask, rotation_deg=rotation_deg[r])

                    # Move back to (h, w, channel)
                    image_transformed = np.moveaxis(image_transformed, 0, 2)
                    mask_transformed = np.moveaxis(mask_transformed, 0, 2)
                    lacken_mask_transformed = np.moveaxis(lacken_mask_transformed, 0, 2)

                    # Prepare file names
                    out_file = f'_rotate_{rotation_deg[r]:+02d}-xflip_{X_flip[x]}-yflip_{Y_flip[y]}'
                    img_name = os.path.basename(image_files[i]).split('.')[0] + out_file + '.tif'

                    # Save transformed images and masks
                    image_save_path = os.path.join(save_folder, images_folder, img_name)
                    mask_save_path = os.path.join(save_folder, mask_folder, img_name)
                    lacken_mask_save_path = os.path.join(save_folder, lacken_mask_folder, img_name)

                    Image.fromarray(image_transformed[:, :, 0]).save(image_save_path)
                    Image.fromarray(mask_transformed[:, :, 0]).save(mask_save_path)
                    Image.fromarray(lacken_mask_transformed[:, :, 0]).save(lacken_mask_save_path)

                    print(
                        f'Saved augmented data for image {i} to: {image_save_path}, {mask_save_path}, {lacken_mask_save_path}')
