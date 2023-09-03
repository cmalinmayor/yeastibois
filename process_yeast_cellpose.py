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

logging.basicConfig(level=logging.INFO)
logging.getLogger("cellpose").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def proc_cellpose_zarr(
    path,
    gpu=True,
    channel=None,
    model="cyto2",
    channel_axis=None,
    diameter=None,
    do_3D=False,
    anisotropy=None,
    cellprob_threshold=None,
):
    """Summary: creates masks with cellpose and fills mask directory in zarr.

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
    ## open zarr
    new_filename = path.split(".")[0] + "_RawMask.zarr"

    zarr_path = path

    ## create new zarr named "new_filename", with raw and masks folder
    zarr_file = zarr.open(zarr_path, mode="r")
    new_zarr = zarr.open(
        new_filename, "a"
    )  ## a will create if it doesn't exist (r+ will not). This zarr is now open

    # fill new zarr raw file with images and create empty mask
    logger.info(f"Copying raw data into new .zarr.")
    new_zarr.require_dataset(
        "raw", data=zarr_file, shape=zarr_file.shape, dtype=zarr_file.dtype
    )  ## this will not override current data, if there is anything there
    new_zarr.create_dataset(
        "masks",
        shape=(zarr_file.shape[0],) + zarr_file.shape[2:],
        dtype=zarr_file.dtype,
        overwrite=True,
    )

    ## open new zarr raws and masks folders
    raw = new_zarr["raw"]
    masks = new_zarr["masks"]

    ## loop through time frames of all images in .zarr raw folder
    logger.info(f"Starting to generate masks")

    for frame in tqdm(range(raw.shape[0])):
        frame_mask = run_cellpose(
            image=raw[frame],  # raw of time 'frame' and z stack 'z'
            gpu=gpu,
            channel=channel,
            model=model,
            channel_axis=channel_axis,
            diameter=diameter,
            do_3D=do_3D,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
        )
        logger.info(f"Current frame is {frame} of {raw.shape[0]}")
        logger.info(frame_mask.dtype)
        logger.info(frame_mask.shape)
        masks[frame] = frame_mask  # add mask to mask folder in zarr file


## remember to set c_axis to 1 when calling, and channel = [1,0] ?


def proc_cellpose_tif(
    path,
    gpu="False",
    channel=None,
    model="cyto2",
    channel_axis=None,
    diameter=None,
    do_3D=False,
    anisotropy=None,
    cellprob_threshold=None,
):
    """Summary: create masks from tiff files with cellpose.

    Args:
        path (str) = path to file
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
    images = load_baby_yeast(path)

    img_outdir = Path(path) / "imgs"
    img_outdir.mkdir(exist_ok=True)
    mask_outdir = Path(path) / "masks"
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


# click options for main()
# @click.command()
# @click.option(
#     "--use_zarr", default=True, help="True if data is stored as zarr. False if tif."
# )
# @click.argument("path")
# @click.option("--gpu", default=False)
# @click.option(
#     "--channel",
#     help="list of channels, either of length 2 or of length number of images by 2. Default is None.",
# )
# @click.option("--model")
# @click.option("--channel_axis")
# @click.option("--diameter")
# @click.option("--do_3D")
# @click.option("--anisotropy")
# @click.option("--cellprob_threshold")
def main(
    path: str,
    use_zarr: bool = True,  ## default is Zarr files
    gpu: bool = True,
    channel_seg: int = 0,  ## when calling in command line, need to use --channel-seg with no underscore.
    channel_nuc: int = 0,
    model: str = "nuclei",
    channel_axis: int = 0,
    diameter: int = 15.0,
    do_3D: bool = True,
    anisotropy: float = 3,
    cellprob_threshold: float = 3.0,
):
    channel = [(channel_seg, channel_nuc)]

    if use_zarr:
        proc_cellpose_zarr(
            path=path,
            gpu=gpu,
            channel=channel,
            model=model,
            channel_axis=channel_axis,
            diameter=diameter,
            do_3D=do_3D,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
        )

    else:
        proc_cellpose_tif(
            path=path,
            gpu=gpu,
            channel=channel,
            model=model,
            channel_axis=channel_axis,
            diameter=diameter,
            do_3D=do_3D,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
        )


if __name__ == "__main__":
    typer.run(main)
