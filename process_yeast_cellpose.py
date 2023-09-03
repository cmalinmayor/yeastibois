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


def proc_cellpose_zarr(
    zarr_file,
    gpu="False",
    channel=None,
    model="cyto2",
    channel_axis=None,
    diameter=None,
    do_3D=False,
    anisotropy=None,
):
    """Summary: fill mask directory in zarr, with masks of raw images created with cellpose.

    Args:
        zarr_file (str) = zarr containing masks and images datasets.
        gpu: Whether or not to use GPU, will check if GPU available.
            Default is set to False.
        channel:
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        model (str):
            The model type that you want to run. Default model is cyto.
            Alternatives from cellpose: "cyto", "nuclei", "tissuenet", "cyto2", and more.
            See Cellpose documentation: https://cellpose.readthedocs.io/en/latest/models.html
            Otherwise, can be a custom cellpose model trained with additional data.
        channel_axis (int):
            An optional parameter to specify which dimension in numpy array contains Channels.
            Default None.
        diameter (int):
            Approximate cell diameter in pixels, if None, default value in model.eval() is 30
        do_3D (bool):
            True if you are creating masks from a 3D dataset. Default set to False.

    Returns:
        masks (np.array) = a segmented mask object for the file
    """

    with zarr.open(zarr_file, "r+") as zarr_root:
        input_data = zarr_root[raw]  # 4d array

        zarr_root.create_dataset(output_group_name, shape=input_data.shape)

        for frame in range(input_data.shape[0]):  ## loop through images in .zarr
            for z in range(input_data.shape[1]):  ## loop through z-stacks in image
                frame_mask = run_cellpose(
                    input_data[frame, z],
                    model=model,
                )
                zarr_root["mask"][frame, z] = frame_mask


path = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/4dmz_2min_40x_01007.zarr"



def proc_cellpose_tif(
    directory,
    gpu="False",
    channel=None,
    model="cyto2",
    channel_axis=None,
    diameter=None,
    do_3D=False,
    anisotropy=None,
):
    """Summary: create masks from tiff files

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
        masks (np.array) = a segmented mask object for the file
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


# if __name__ == "__main__":
#     masks = run_cellpose_yeast(
#         directory="/mnt/efs/shared_data/YeastiBois/baby_data/Fig1_brightfield_and_seg_outputs",
#         gpu=True,
#         channel=None,
#         model="cyto2",
#         channel_axis=None,
#         diameter=30,
#         do_3D=True,
#         anisotropy=3,
#     )
