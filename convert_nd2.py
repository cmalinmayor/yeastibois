
# %%
import os
from glob import glob

import matplotlib.pyplot as plt

import nd2
import numpy as np
import zarr

#%%

#import tifffile as tiff
#from pims import ND2_Reader


# input_nd2_path = '../data/4dmz_2min_40x_01007.nd2'
input_nd2_path = '../4dmz_2min_40x_01007.nd2'
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
    ax[i,0].set_title(f'z = {i}')
plt.show()

# %%
# arr.to_zarr('../data/4dmz_2min_40x_01007.zarr')
arr.to_zarr('../convertedzarr/4dmz_2min_40x_01007.zarr')

# %% This is how I get the percentiles
np.percentile(arr, [5, 95])

#%% save normalized
percentiled = np.percentile(arr, [5, 95])
percentiled.shape
# normarr = arr.clip(a_min=amin, a_max=amax)



# %% This is how I plot using the percentiles
fig, ax = plt.subplots(10, 2, figsize=(10, 50))
for i in range(10):
    ax[i,1].imshow(arr[0,i,1], vmin=50, vmax=350)
    ax[i,0].imshow(arr[0,i,0], vmin=50, vmax=350)
plt.show()

#%%
fig,ax = plt.subplots(25,2,figsize=(10,100))
for i in range(25):
    ax[i,0].imshow(arr[i,2,0,:,:], vmin=50, vmax=350)
    ax[i,1].imshow(arr[i,2,1,:,:], vmin=50, vmax=350)
    ax[i,0].set_title(f't = {i}')

plt.show()



# %% save images
import Image
for i in range(25):
    fig = plt.imshow(arr[i,2,0,:,:],vmin=50, vmax=350)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(f'../convertedtif/t_{i}_c_{z}_4dmz.tif',bbox_inches='tight')
    break

plt.imshow(f'../convertedtif/t_{i}_c_{z}_4dmz.tif')

#%% move images files into appropriate subdirectory
import os
import shutil

source_directory = '../convertedtif/'
actin_directory = '../convertedtif/actin_all/'
half_labeled_directory = '../convertedtif/half_labeled/'

# Create the target directories if they don't exist
os.makedirs(actin_directory, exist_ok=True)
os.makedirs(half_labeled_directory, exist_ok=True)

# List all files in the source directory
files = os.listdir(source_directory)

# Define the subdirectory names
subdirectory_names = {'0': actin_directory, '1': half_labeled_directory}

# Iterate through the files and move them to the appropriate subdirectory
for filename in files:
    if filename.endswith('.tif'):
        parts = filename.split('_')
        if parts[3]=='0':
            file = os.path.join(source_directory,filename)
            shutil.move(file,actin_directory)
        elif parts[3] == '1':
            file = os.path.join(source_directory,filename)
            shutil.move(file,half_labeled_directory)
        else:
            print('broke: ',filename)
print('all files processed')
