# %%
from cellpose import models
import numpy as np
import matplotlib.pyplot as plt
from process_zebrafish_zarr import process_bette_zarr
import zarr

#%% preprocess dataset into groups
# zarr_file = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023.zarr"
# process_bette_zarr(zarr_file)

# # %% Get voxel size to compute anisotropy
# import nd2
# nd2_data = "/mnt/efs/shared_data/YeastiBois/TiensVideos/glass_60x_023.nd2"
# f = nd2.ND2File(nd2_data)
# f.voxel_size()

# %% Load data
zarr_file = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_processed.zarr"
zfile = zarr.open(zarr_file)

# %% T, Z, C, X, Y (15, 11, 3, 1024, 1024)
data = zfile['exp1']

# %%
def pretty_plot(image): 
    p5, p95 = np.percentile(image, [1, 99])
    plt.imshow(image, cmap='gray', vmin=p5, vmax=p95)
    plt.show()

from skimage.color import label2rgb

def pplot_img_mask(image, mask, figsize=(10, 5)):
    p5, p95 = np.percentile(image, [1, 99])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image, cmap='gray', vmin=p5, vmax=p95)
    ax2.imshow(label2rgb(mask))
    ax1.axis('off')
    ax2.axis('off')
    plt.show()
    
# %% Plot some examples
pretty_plot(data[0, 0, 0])
pretty_plot(data[0, 0, 1])
pretty_plot(data[0, 0, 2])
# %%
model = models.Cellpose(model_type='cyto2', gpu=True)

# %%
membrane_idx = 0
nucleus_idx = 2
masks, flows, styles, diams = model.eval(
    data[0], 
    anisotropy=10, 
    diameter=120, 
    channels=[membrane_idx + 1, nucleus_idx + 1], 
    channel_axis=1, 
    do_3D=True
)
# %% Show an example
idx = 7
pplot_img_mask(data[0, idx, membrane_idx], masks[idx])
# %%
