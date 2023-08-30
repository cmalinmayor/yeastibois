import numpy as np
import time, os, sys
from cellpose import models, utils, io
from run_cellpose import run_cellpose

def load_data(directory):
    ''' Load all of the images in the directory into a np array.
    Args:
        directory: The folder containing the images.
    
    Returns:
        images (list): A sorted list of the images in the directory
    '''

    images = cellpose.io.get_image_files(directory)

    #for file in os.listdir(directory):
    #    if file.endswith('.tiff'):
            
    return images


## function for running for loop on yeast
def run_cellpose_yeast(file, directory,
                   gpu='False', channel=None, 
                   model='cyto', channel_axis=None):
    '''create masks from tiff files

    Args:
        file (str) = a .tiff file
        directory (str) = path to file

    Returns:
        mask (np.array) = a segmented mask object for the file
    '''

    # calls load to tiff
    images = load_data(directory)

    # calls run_cellpose
    masks = []
    for image in images:
        mask = run_cellpose(image, gpu='False', channel=None, model='cyto', channel_axis=None)
        masks.append(mask)

    return masks
