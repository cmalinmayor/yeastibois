#tracking

#%% imports
import zarr
import numpy as np
import pandas as pd
from motile.plot import draw_track_graph, draw_solution
import motile
import networkx as nx
import napari
import skimage

import os
# set environment variable
os.environ['DISPLAY'] = ':1'
# now Qt knows where to render
# viewer = napari.Viewer()

#%% import data
# zarrpath = 'C:/Users/TienC/Documents/CellTrackingTesting/micro-sam/c1z5.zarr/'

zarrpath = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_RawMasks.zarr"
zarrfile =  zarr.open(zarrpath,'r')
mask = zarrfile['masks'] #segmentation mask
raw = zarrfile['raw']
unique = np.unique(mask) #number of unique labeles in the segmentation mask
# fullmask = mask[:] #load the full mask into memory
nonzero_unique = unique[1:] #zero is empty space

#%% napari plot
raw_frame = raw[0, :, 2]
mask_frame = mask[0].astype(int)
mask_clean = skimage.morphology.remove_small_objects(mask_frame, min_size=5000)
maskiter = np.zeros(mask_frame.shape)
for i in range(maskiter.shape[0]):
    maskiter[0] = skimage.morphology.remove_small_objects(mask_frame, min_size=1000)

print(len(np.unique(mask_clean)))


regions = skimage.measure.regionprops(mask_frame)
centers = [r.centroid for r in regions]


#%%
NapariViewer = napari.Viewer()
NapariViewer.add_image(raw,name='data')
NapariViewer.add_labels(mask.astype(int),name='labels')

#%%
for l in NapariViewer.layers:
    l.scale = [1, 10, 1, 1]

# %%
