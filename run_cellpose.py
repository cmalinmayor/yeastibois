import numpy as np
import time, os, sys
from cellpose import models, utils, io


def run_cellpose(image, gpu='False', channel=None, model='cyto', channel_axis=None):
    '''
    ARGS:
        image: a 2d numpy array (w, h). It will contain a grey-scale image. 
        gpu: Whether or not to use GPU, will check if GPU available. 
            Default is set to False.
        channel: list of channels, either of length 2 or of length number of images by 2. 
            First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). 
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue). 
        model: The model type that you want to run. Default model is cyto. 
            Alternatives from cellpose: "cyto", "nuclei", "tissuenet", "cyto2", and more. 
            See Cellpose documentation: https://cellpose.readthedocs.io/en/latest/models.html 
            Otherwise, can be a custom cellpose model trained with additional data.
        channel_axis: An optional parameter to specify which dimension in numpy array contains Channels. 
            Default None.

    RETURNS:
        mask: the function returns a mask containing instance segmentations of the image. It is a 2d array (w, h).

    '''
    # create model
    cellpose_model = models.Cellpose(gpu=gpu, model_type=model)

    # create masks from image
    masks, flows, styles, diams = model.eval(image, diameter=None, channels=channel, channel_axis=channel_axis)

    return masks


## function to create array from tiff, and from zarr (load_data.py) two functions

## function for running for loop on yeast (process_yeast_cellpose.py)
    # calls load to tiff
    # calls run_cellpose


## function for running for loop on tiffs

## function for running for loop on 3d data set