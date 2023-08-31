import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


## function to create array from tiff
def load_tiff(file):
    '''
    ARGS:
        file: a tiff file

    RETURNS:
        image: a numpy array

    '''
    image = plt.imread(file)    
    return image


def load_3d_tiff(list_of_files):
    """Load a list of tiff files into a 3d array. Assumes each file is a z slice
    and they are in order.

    Args:
        list_of_files (_type_):filenames of z slices to be combined into one array

    Returns:
        np.array: 3d numpy array (z, _, _)
    """
    images = []
    for file in list_of_files:
        images.append(load_tiff(file))
    return np.array(images)


#Creating a for loop to iterate over a z-stack
def run_cellpose_z_slices(z_stack):
    """ Run cellpose on multiple z slices and return masks

    Args:
        z_stack (np.array): 3D array with z as first dimension

    Returns:
        np.array: one mask per z slice
    """
    zstack_masks = []
    for i in z_stack:
        image = load_tiff(i)
        mask = run_cellpose(image)
        zstacks_masks.append(mask)
    return zstack_masks
        
        
        
## function to create array from zarr