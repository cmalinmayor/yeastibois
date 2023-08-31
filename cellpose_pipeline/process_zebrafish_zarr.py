import os

import np
import zarr

from run_cellpose import run_cellpose


def process_zarr(zarr_file, input_group_name, output_group_name, gpu=False):
    """Process zarr z-slices with cellpose and save masks to another channel in the zarr.

    Args:
        zarr_file (_type_): Contains a 3D + t zebrafish data. Assume dimensions are
            (t, z, y, x). 
        input_group_name (str): usually "gfp"
        output_group_name (str): zarr group to save the output masks to
        gpu (bool): True if you want to use gpu. defaults to False
    """
    data = zarr.open(zarr_file, 'r+')
    input = data[input_group_name]  # 4d array
    masks = []
    for frame in input:
        frame_masks = []
        for z in frame:
            frame_masks.append(run_cellpose(input[frame, z], model='nuclei'))
        masks.append(frame_masks)
    data.create_group(output_group_name, shape=masks.shape)
    data[output_group_name] = np.array(masks)