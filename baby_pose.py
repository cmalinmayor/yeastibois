# %%
from cellpose import models
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from imageio import imread
from skimage.color import label2rgb
import zarr
from iou import predict
from scipy.stats import mode
import copy

# %% Utility functions for plotting images
def pretty_plot(image): 
    p5, p95 = np.percentile(image, [1, 99])
    plt.imshow(image, cmap='gray', vmin=p5, vmax=p95)
    plt.show()


def pplot_img_mask(image, mask, figsize=(5, 5), ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    # ax1.imshow(image, cmap='gray', vmin=p5, vmax=p95)
    ax.imshow(label2rgb(mask, image=image, bg_label=0))
    ax.axis('off')

def plot_zstack(img_stack, mask_stack):
    n_z = len(img_stack)
    fig, axes = plt.subplots(1, n_z, figsize=(15, 3))
    for i in range(n_z):
        pplot_img_mask(img_stack[i], mask_stack[i], ax=axes[i])



if __name__ == "__main__":
    # %% Load a trap
    # Define top-level directory and names of Z section sub-directories
    base_img_dir = Path('/mnt/efs/shared_data/YeastiBois/baby_data/Fig1_brightfield_and_seg_outputs')
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
    bfimgs = np.stack(bfimgs, axis=1)

    print('Shape of "bfimgs":', bfimgs.shape)


    # %%
    model = models.Cellpose(model_type='nuclei', gpu=True)

    # %% TODO make this tp by tp, all stacks, possibly use 3d
    all_z_masks = []
    for z in range(5):
        data = bfimgs[:, z]
        masks, flows, styles, diams = model.eval(
            [x for x in data], 
            diameter=8, 
        )
        all_z_masks.append(masks)

    # %%
    all_masks = np.stack(all_z_masks, axis=1)

    # %%
    # %%
    for i in range(15):
        plot_zstack(bfimgs[i], all_masks[i])

    # %%
    # Greedy match the objects in the z-stacks based on IOU
    # This essentially is tracking, and uses a linear assignment problem

    for i in range(10, 20):
        labels = predict(all_masks[i])
        labels[0] = all_masks[i, 0]
        ignore = labels != 0
        m, _= mode(np.where(ignore, labels, np.NaN), axis=0)
        m, _ = mode(labels, axis=0, nan_policy='omit')
        pplot_img_mask(bfimgs[i, 0], m)

    # %%
    for j in range(10, 15):
        labels = all_masks[j] #predict(all_masks[j])
        tracked = predict(all_masks[j])
        fig, axes = plt.subplots(2, 6, figsize=(17, 6))
        for i in range(5):
            axes[0][i].imshow(
                labels[i], 
                vmin=0, 
                vmax=labels.max(), 
                cmap='tab10', 
                interpolation='none'
            )
            axes[1][i].imshow(
                tracked[i], 
                vmin=0, 
                vmax=tracked.max(), 
                cmap='tab10', 
                interpolation='none'
            )
        axes[0][5].imshow(labels.max(axis=0))
        axes[1][5].imshow(tracked.max(axis=0))
    # %%
    from tqdm import tqdm
    all_tracked = []
    for j in tqdm(range(300)):
        all_tracked.append(predict(all_masks[j]).max(axis=0))

    # %%
    fig, axes = plt.subplots(10, 5, figsize=(15, 30))
    for i, ax in enumerate(axes.ravel()):
        pplot_img_mask(bfimgs[i, 2], all_tracked[i], ax=ax)
    fig.tight_layout()

    # %%
    tracks = np.stack(all_tracked)
    save_file = "/mnt/efs/shared_data/YeastiBois/baby_pose.zarr"
    zfile = zarr.open(save_file)
    zfile['detections'] = tracks


    # %%
