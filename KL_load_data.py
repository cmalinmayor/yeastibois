from pathlib import Path
import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
import os
from imageio import imread
import tifffile
from tqdm import tqdm
import zarr


def load_zarr_data(directory, file, gpu="False"):
    """Summary: Open zarr and create a new zarr containing the original images (saved in 'raw') and an empty mask zarr.

    Args:
        directory (str):
            the path to the zarr containing images.

        file (_type_):
            the zarr file itself. Contains images, assume dimensions are (t, z, y, x).

        gpu (bool):
            True if you want to use gpu. defaults to False

    Returns:
        Creates new zarr file with a raw zarr, containing all images, and an empty mask zarr of the same shape.
        No return.
    """

    new_filename = file.split(".")[0] + "_RawMask" + ".zarr"
    zarr_path = os.path.join(directory, file)

    zarr_file = zarr.open(zarr_path, mode="r")
    new_zarr = zarr.open(new_filename, "w")
    new_zarr.create_dataset("raw", data=zarr_file)
    new_zarr.create_dataset("mask", shape=zarr_file.shape)
    print(
        "Completed creating new zarr file with datasets for raw and mask. Raw contains images. Mask dataset is empty."
    )

# if __name__ == "__main__":
# load_raw_masks(
#     "/mnt/efs/shared_data/YeastiBois/baby_data/Fig1_brightfield_and_seg_outputs"
# )



def load_tiff_raw_masks(directory, file_type):
    """Summary: Load raw files and mask files from a folder, and return as numpy arrays.
    Args:
        directory (str):
            the path to the folder containing images and masks, each stored as .tiffs.

    Returns:
        raws_np:
            Numpy array storing the raw images.
        masks_np:
            Numpy array storing the segmented masks.
    """

    path_raw = Path(directory) / "imgs"
    path_masks = Path(directory) / "masks"
    print(path_raw)

    # loop through images folder, save them in a numpy array
    raws_np = []
    for i, raw in tqdm(
        enumerate(sorted(path_raw.iterdir()))
    ):  # i is a counter for our loop
        ## iterdir does not keep the order of the files, so need to sort. alternative: use glob.
        raws_np.append(tifffile.imread(raw))

    # loop through masks folder, save them in a numpy array
    masks_np = []
    for i, mask in tqdm(
        enumerate(sorted(path_masks.iterdir()))
    ):  # i is a counter for our loop
        masks_np.append(tifffile.imread(mask))

    return raws_np, masks_np



## function to load tiffs (specific from baby data)
def load_baby_yeast(directory):
    """Load all of the images in the directory into a np array.
    Args:
        directory (str):
            The folder containing the images.

    Returns:
        bfimgs (np.array of dim (t, w, h, z)):
            All Z-sections together as a N_timepoint * image_width * image_height * N_zsections array
    """

    # Define top-level directory and names of Z section sub-directories
    base_img_dir = Path(directory)
    z_dir_names = [f"brightfield_z{i}" for i in range(1, 6)]

    # Load images for each Z section in a loop
    bfimgs = []
    for z in z_dir_names:
        # Get a list of all png files in the Z-section sub-directory
        zfiles = filter(lambda x: x.suffix == ".png", (base_img_dir / z).iterdir())
        # Sort by file name to guarantee ordering by time point
        zfiles = sorted(zfiles, key=lambda x: x.stem)
        # Load all files and stack them to form an N_timepoint * image_width * image_height array
        bfimgs.append(np.stack([imread(f) for f in zfiles]))

    # Stack all Z-sections together to form an N_timepoint * image_width * image_height * N_zsections array
    bfimgs = np.stack(bfimgs, axis=-1)
    bfimgs = np.transpose(bfimgs, (0, 3, 1, 2))

    print('Shape of "bfimgs":', bfimgs.shape)

    return bfimgs


def load_tiff(file):
    """ Summary: function to create array from tiff
    ARGS:
        file: a tiff file
        directory: the path of the tiff file

    RETURNS:
        image: a numpy array

    """
    image = plt.imread(file)

    return image



def load_tiff_zstack(file):
    '''
    Args:
        file: tiff file.
    Returns: 
        zstack_masks: a list of masks created from cellpose on an image. 
    '''
    zstack_masks = []
    for i in file:
        image = load_tiff(i)
        mask = run_cellpose(image)
        zstacks_masks.append(mask)
    return zstack_masks
