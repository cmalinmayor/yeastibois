import numpy as np
import time
import os
import sys
from cellpose import models, utils, io
from run_cellpose import run_cellpose
from matplotlib import pyplot as plt
from pathlib import Path
from imageio import imread
from load_data_cellpose import load_data_yeast
import tifffile
import logging
from skimage.color import label2rgb
from tqdm import tqdm


def pretty_plot(image):
    p5, p95 = np.percentile(image, [1, 99])
    plt.imshow(image, cmap="gray", vmin=p5, vmax=p95)
    plt.show()


def pplot_img_mask(image, mask, figsize=(10, 5)):
    p5, p95 = np.percentile(image, [1, 99])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image, cmap="gray", vmin=p5, vmax=p95)
    ax2.imshow(label2rgb(mask))
    ax1.axis("off")
    ax2.axis("off")
    plt.show()


# function for running for loop on yeast


def run_cellpose_yeast(
    directory,
    gpu="False",
    channel=None,
    model="cyto2",
    channel_axis=None,
    diameter=None,
    do_3D=False,
    anisotropy=None,
):
    """create masks from tiff files

    Args:
        directory (str) = path to file
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
        diameter: approximate cell diameter in pixels, if None, default value in model.eval() is 30

    Returns:
        mask (np.array) = a segmented mask object for the file
    """
    # calls load to tiff
    images = load_data_yeast(directory)

    img_outdir = Path(directory) / "imgs"
    img_outdir.mkdir(exist_ok=True)
    mask_outdir = Path(directory) / "masks"
    mask_outdir.mkdir(exist_ok=True)

    # calls run_cellpose
    masks = []
    for i, image in tqdm(enumerate(images)):  # i is a counter for our loop
        mask = run_cellpose(
            image,
            gpu=gpu,
            channel=channel,
            model=model,
            channel_axis=channel_axis,
            diameter=diameter,
            do_3D=do_3D,
            anisotropy=anisotropy,
        )
        # create mask tiff file, named with counter
        tifffile.imwrite(img_outdir / f"raw{i:03d}.tif", image)
        tifffile.imwrite(mask_outdir / f"mask{i:03d}.tif", mask)
        masks.append(mask)

    return masks


if __name__ == "__main__":
    masks = run_cellpose_yeast(
        directory="/mnt/efs/shared_data/YeastiBois/baby_data/Fig1_brightfield_and_seg_outputs",
        gpu=True,
        channel=None,
        model="cyto2",
        channel_axis=None,
        diameter=30,
        do_3D=True,
        anisotropy=3,
    )
