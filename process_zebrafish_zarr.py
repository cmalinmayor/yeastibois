import os
import numpy as np
import zarr
from cellpose import models, utils, io
from run_cellpose import run_cellpose


def process_zebrafish_zarr(zarr_file, input_group_name, output_group_name, gpu=False):
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
    data.create_group(output_group_name, shape=input.shape)
    for frame in range(data.shape[0]):
        for z in range(data.shape[1]):
            frame_mask = run_cellpose(input[frame, z], model='nuclei')
            data[output_group_name][frame, z] = frame_mask


def process_bette_zarr(zarr_file):
    """Process zarr z-slices with cellpose and save masks to a new zarr.

    Args:
        zarr_file (_type_): Contains a 3D + t +c frog data. Assume dimensions are
            (t, experiment, z, c, x, y). :(
        input_group_name (str): usually "gfp"
        output_group_name (str): zarr group to save the output masks to
        gpu (bool): True if you want to use gpu. defaults to False
    """

    input_data = zarr.open(zarr_file, 'r') # 6d array
    print('input data shape = ',input_data.shape)
    processed_zarr_name = zarr_file.split('.zarr')[0] + '_processed' + '.zarr'
    print('new zarr name = ',processed_zarr_name)
    useful_data = zarr.open(processed_zarr_name,'w')
    useful_data.create_group('exp0')
    useful_data['exp0'] = input_data[:,0]
    useful_data.create_group('exp1')
    useful_data['exp1'] = input_data[:,1]
    print('reshaped 6dimensional zarr')
    return

    # # useful_data = input_data[:,1,:,1]  # use experiment 1, channel 1 -> 4D array
    # print('useful data shape = ',useful_data.shape)
    # output = zarr.save(shape=useful_data.shape) # this is not code
    # for frame in range(useful_data.shape[0]):
    #     for z in range(useful_data.shape[1]):
    #         frame_mask = run_cellpose(input[frame, z], model='cyto')
    #         output[frame, z] = frame_mask