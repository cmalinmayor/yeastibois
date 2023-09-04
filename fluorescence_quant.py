import numpy as np
import time
import os
import sys
from cellpose import models, utils, io
from run_cellpose import run_cellpose
from matplotlib import pyplot as plt
from pathlib import Path
from imageio import imread
import tifffile
import logging
from skimage.color import label2rgb
from tqdm import tqdm
import zarr
import typer

def fluor_quant(mask, raw_fluor):
    
    '''
    Summary: This function returns single-cell fluoresence intensities across time from tracked cells.

    Args:
        Mask (np.array): A folder of masks
        Raw_fluor (np.array): A folder of images, from which we will pull the fluorescence channel. 

    '''

    tracked_data = mask #your tracked data, shape=(t,z,y,x)
    flourescence = raw

    for i in tracked_data.shape[0]: #iterate through all timepoints
        regions = skimage.measure.regionprops(tracked_data[i],intensity_image=a)
            for r in regions:
                intensity = r.intensity_mean