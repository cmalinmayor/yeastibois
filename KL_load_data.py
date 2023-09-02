from pathlib import Path
import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
import os
from imageio import imread
import tifffile


def load_raw_masks(directory):
    """
    Args:
        directory: the Path to the folder containing images and masks.

    Returns:
        raws_np: Numpy array storing the raw images.
        masks_np: Numpy array storing the segmented masks.
    """

    path_raw = Path(directory) / "imgs"
    path_masks = Path(directory) / "masks"

    # loop through images folder, save them in a numpy array
    raws_np = []
    for i, raw in tqdm(enumerate(path_raw)):  # i is a counter for our loop
        raws_np.append = tifffile.imread(raw)

    # loop through masks folder, save them in a numpy array
    masks_np = []
    for i, mask in tqdm(enumerate(path_raw)):  # i is a counter for our loop
        masks_np = raws_np.append = tifffile.imread(mask)

    return raws_np, masks_np


load_raw_masks(
    "/mnt/efs/shared_data/YeastiBois/baby_data/Fig1_brightfield_and_seg_outputs"
)
