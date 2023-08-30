import numpy as np
import time, os, sys
from cellpose import models, utils, io


## function for running for loop on yeast (process_yeast_cellpose.py)
def yeast_cellpose(file, path,
                   gpu='False', channel=None, 
                   model='cyto', channel_axis=None):
    '''
    ARGS:
        file = a .tiff file
        path = path to file
        
    RETURNS:
        mask = a segmented mask object for the file
    '''
    # calls load to tiff
    image = load_tiff(file)

    # calls run_cellpose
    mask = run_cellpose(image, gpu='False', channel=None, model='cyto', channel_axis=None)

    return mask
