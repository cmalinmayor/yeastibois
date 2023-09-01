def parse_filename(filename):
    """Get the time point and the channel from the filename of Jose's zebrafish tiffs.

    Args:
        filename (str): a string filename of a single timepoint and channel

    Returns:
        (time, channel) where time is an int and channel is an int
    """
    split_filename = filename.split('_')
    time_section = split_filename[1]
    channel_section = split_filename[6]
    time = int(time_section.split('0')[-1])
    channel = int(channel_section.split('0')[-1])
    return time, channel
# print(parse_filename("S000_t000005_V000_R0000_X000_Y000_C02_I0_D1_P00132.tif"))