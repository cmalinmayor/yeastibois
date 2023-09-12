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
import json


def fluor_quant(mask, raw_fluor):
    """
    Summary: This function returns single-cell fluoresence intensities across time from tracked cells.

    Args:
        Mask (str): path to a zarr array, segmentation, ids of cells consistent over time = after tracking.
        Raw_fluor (str): path to a zarr array, fluo channel.

    """

    tracked_data = mask  # your tracked data, shape=(t,z,y,x)
    flourescence = raw

    out = {}
    for i in tracked_data.shape[0]:  # iterate through all timepoints
        out_t = {}
        regions = skimage.measure.regionprops(
            tracked_data[i], intensity_image=flourescence[i]
        )
        for r in regions:
            intensity = r.intensity_mean
            out_t[r.label] = intensity
        out[i] = out_t

    with open("fluo_intensities.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    f = Path("example.zarr")
    mask = zarr.ooen(f / "tracked_segmentations", "r")
    fluo = zarr.open(f / "fluorescence", "r")
    # select some channel
    print(fluo.shape)
    # all timepoints
    fluo = fluo[:, 1]

    fluor_quant(mask, fluo)
