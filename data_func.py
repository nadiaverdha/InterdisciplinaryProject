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
        output_path = output_directory + '/' +type + '_' + str(row.name) + '_merged.tiff'
        command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]} {row.filepath[2]} {row.filepath[3]}"
        os.system(command)



# to merge the files and split into train, validation and test
def merge_files_and_split(df, type='VV'):
    grouped = df[df['filepath'].str.endswith('.tif')]
    grouped.loc[:, 'date'] = grouped['time'].dt.date
    grouped_type = grouped[grouped['filepath'].str.contains('VH')]
    grouped_type = grouped_type.groupby('date').agg({'filepath': list})
    for _, row in grouped_type.iterrows():
        if (_.month == 4) | (_.month == 7):
            directory = 'test'
            os.makedirs(directory, exist_ok=True)
            output_path = directory + '/' + type + '_' + str(row.name) + '_merged.tiff'
            command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]} {row.filepath[2]} {row.filepath[3]}"
            os.system(command)
        elif (_.month == 5) | (_.month == 8):
            directory = 'val'
            os.makedirs(directory, exist_ok=True)
            output_path = directory + '/' + type + '_' + str(row.name) + '_merged.tiff'
            command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]} {row.filepath[2]} {row.filepath[3]}"
            os.system(command)
        else:
            directory = 'train'
            os.makedirs(directory, exist_ok=True)
            output_path = directory + '/' + type + '_' + str(row.name) + '_merged.tiff'
            command = f"gdal_merge.py -o {output_path} {row.filepath[0]} {row.filepath[1]} {row.filepath[2]} {row.filepath[3]}"
            os.system(command)