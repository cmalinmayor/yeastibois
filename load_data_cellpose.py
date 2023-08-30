import numpy as np
import time, os, sys
import matplotlib.pyplot as plt


## function to create array from tiff
def load_tiff(file):
    '''
    
    ARGS:
        file: a tiff file
        directory: the path of the tiff file 

    RETURNS:
        image: a numpy array

    '''
    image = plt.imread(file)
        

    return image

#Creating a for loop to iterate over a z-stack
def load_tiff_zstack(file):
    zstack_masks = []
    for i in file:
        image =load_tiff(i)
        mask =run_cellpose(image)
        zstacks_masks.append(mask)
    return zstack_masks
        
        
        
## function to create array from zarr