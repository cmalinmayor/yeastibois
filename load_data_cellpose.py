import numpy as np
import time, os, sys
import matplotlib.pyplot as plt


## function to create array from tiff
def load_tiff(file):
    '''
    
    ARGS:
        file: a tiff file
        directory: the path of the tiff file 

    RETURNS:
        image: a numpy array

    '''
    image = plt.imread(file)
        

    return image

#Creating a for loop to iterate over a z-stack
def load_tiff_zstack(file):
    zstack_masks = []
    for i in file:
        image =load_tiff(i)
        mask =run_cellpose(image)
        zstacks_masks.append(mask)
    return zstack_masks
        
        
        
## function to create array from zarr

## function to load tiffs (specific from baby data)
def load_data_yeast(directory):
    ''' Load all of the images in the directory into a np array.
    Args:
        directory (str): 
            The folder containing the images.
    
    Returns:
        bfimgs (np.array of dim (t, w, h, z)): 
            All Z-sections together as a N_timepoint * image_width * image_height * N_zsections array
    '''

    #images = cellpose.io.get_image_files(directory)
    
    # Define top-level directory and names of Z section sub-directories
    #base_img_dir = Path('paper-data/Fig1_brightfield_and_seg_outputs')
    #z_dir_names = [f'brightfield_z{i}' for i in range(1,6)]
    base_img_dir = Path(directory)
    z_dir_names = [f'brightfield_z{i}' for i in range(1,6)]

    # Load images for each Z section in a loop
    bfimgs = []
    for z in z_dir_names:
        # Get a list of all png files in the Z-section sub-directory
        zfiles = filter(lambda x: x.suffix == '.png', (base_img_dir / z).iterdir())
        # Sort by file name to guarantee ordering by time point
        zfiles = sorted(zfiles, key=lambda x: x.stem)
        # Load all files and stack them to form an N_timepoint * image_width * image_height array
        bfimgs.append(np.stack([imread(f) for f in zfiles]))

    # Stack all Z-sections together to form an N_timepoint * image_width * image_height * N_zsections array
    bfimgs = np.stack(bfimgs, axis=-1)

    print('Shape of "bfimgs":', bfimgs.shape)

    return bfimgs