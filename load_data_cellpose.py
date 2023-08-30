import numpy as np
import time, os, sys
import matplotlib.pyplot as plt


## function to create array from tiff
def load_tiff(file, directory="directory"):
    '''
    
    ARGS:
        file: a tiff file
        directory: the path of the tiff file 

    RETURNS:
        image: a numpy array

    '''


    for i in directory:
        path = os.path.join(directory, i)
        image = plt.imread(file)

    return image


## function to create array from zarr