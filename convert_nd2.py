# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from glob import glob

import matplotlib.pyplot as plt
# %%
import nd2
import numpy as np
import zarr

#import tifffile as tiff
#from pims import ND2_Reader


input_nd2_path = '../data/4dmz_2min_40x_01007.nd2'
#my_array = nd2.imread(input_nd2_path)                          # read to numpy array
#my_array = nd2.imread('some_file.nd2', dask=True)               # read to dask array
#my_array = nd2.imread('some_file.nd2', xarray=True)             # read to xarray
#my_array = nd2.imread('some_file.nd2', xarray=True, dask=True)  # read file to dask-xarray

# or open a file with nd2.ND2File
f = nd2.ND2File(input_nd2_path)

# (you can also use nd2.ND2File() as a context manager)
#with nd2.ND2File(input_nd2_path) as ndfile:
 #   print(ndfile.metadata)


# ATTRIBUTES:   # example output
print(f.path)          # 'some_file.nd2'
print(f.shape)        # (10, 2, 256, 256)
print(f.ndim)          # 4
print(f.dtype)         # np.dtype('uint16')
print(f.size)          # 1310720  (total voxel elements)
print(f.sizes)         # {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
print(f.is_rgb )       # False (whether the file is rgb)
                # if the file is RGB, `f.sizes` will have
                # an additional {'S': 3} component

arr = f.to_dask()
print(arr.shape)
fig, ax = plt.subplots(25, 2, figsize=(10, 100))
for i in range(25):
    ax[i,0].imshow(arr[i,5,0])
    ax[i,1].imshow(arr[i,5,1])
plt.show()

# %%
arr.to_zarr('../data/4dmz_2min_40x_01007.zarr')

# %% This is how I get the percentiles
np.percentile(arr, [5, 95])
# %% This is how I plot using the percentiles
fig, ax = plt.subplots(10, 2, figsize=(10, 50))
for i in range(10):
    ax[i,0].imshow(arr[0,i,0], vmin=50, vmax=350)
    ax[i,1].imshow(arr[0,i,1], vmin=50, vmax=350)
plt.show()
