import numpy as np
import time
import os
import sys
from cellpose import models, utils, io
import matplotlib.pyplot as plt


def run_cellpose(
    image,
    gpu,
    channel,
    model,
    channel_axis,
    diameter,
    do_3D,
    anisotropy,
    cellprob_threshold=4,
):
    
    """
    ARGS:
        image (np array):
            a 2d numpy array (w, h). It will contain a grey-scale image.
        gpu (boolean):
            Whether or not to use GPU, will check if GPU available.
        channel (int):
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        model (str):
            The model type that you want to run.
            Alternatives from cellpose: "cyto", "nuclei", "tissuenet", "cyto2", and more.
            See Cellpose documentation: https://cellpose.readthedocs.io/en/latest/models.html
            Otherwise, can be a custom cellpose model trained with additional data.
        channel_axis (int):
            An optional parameter to specify which dimension in numpy array contains Channels.
        diameter (int):
            Approximate cell diameter in pixels. If None, default value in model.eval() is 30
        do_3D (boolean):
            True if you want the model to run on 3D data.
        anisotropy (int):
            Ratio between the x, y an z channel. Only needed for 3D.
        cellprob_threshold (float):(float (optional, default 3.0)) – all pixels with value above
          threshold kept for masks, decrease to find more and larger masks.

    RETURNS:
        mask: the function returns a mask containing instance segmentations of the image. It is a 2d array (w, h).

    """

    # create model
    cellpose_model = models.Cellpose(gpu=gpu, model_type=model)

    # create masks from image
    masks, flows, styles, diams = cellpose_model.eval(
        image,
        diameter=diameter,
        channels=channel,
        channel_axis=channel_axis,
        do_3D=do_3D,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob_threshold,
        net_avg=True,
        resample=True,
        min_size=128,
    )
    
    return masks



