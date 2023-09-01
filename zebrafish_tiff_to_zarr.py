import os

import tiffile
import zarr


def zebrafish_tiff_to_zarr(directory, output_zarr_path):
    """Read in a directory of tiffs, where each tiff is named by time and channels 
    and contains (z y x), and saves the data as a zarr, with a different group for each
    channel, and dimensions (t, z, y, x).

    Args:
        directory (str): directory containing tiffs
        output_zarr_path (str): the path to the output zarr, including the zarr name
    """
    channels = ['gfp', 'rfp']  # this corresponds to the order of the channels.
    time = max([parse_filename(filename)[0]
                for filename in os.listdir(directory)
                if filename.endswith('tif')])
    print(time)
    z = 132  # could read spatial dimension from one tiff
    y = 2048
    x = 2048
    # gpf is channel 1, rfp is channel 2
    with zarr.open(output_zarr_path, 'w') as root:
        for channel in channels:
            root.create_dataset(channel, shape=(time, z, y, x))
        for filename in os.listdir(directory):
            if filename.endswith('tif'):
                data = tiffile.imread(os.path.join(directory, filename))
                time, channel_num = parse_filename(filename)
                channel_name = channels[channel_num-1]
                root[channel_name][time] = data


def parse_filename(filename):
    """Get the time point and the channel from the filename of Jose's zebrafish tiffs.

    Args:
        filename (str): a string filename of a single timepoint and channel

    Returns:
        (time, channel) where time is an int and channel is an int
    """
    print(filename)
    split_filename = filename.split('_')
    time_section = split_filename[1]
    channel_section = split_filename[6]
    time = int(time_section[1:])
    channel = int(channel_section[1:])
    print(time, channel)
    return time, channel


if __name__ == "__main__":
    print(parse_filename("S000_t000005_V000_R0000_X000_Y000_C02_I0_D1_P00132.tif"))
    zebrafish_tiff_to_zarr('/mnt/efs/shared_data/YeastiBois/Jose_zebrafish_lightsheet/tiffs', 
                           '/mnt/efs/shared_data/YeastiBois/Jose_zebrafish_lightsheet/R01.zarr')