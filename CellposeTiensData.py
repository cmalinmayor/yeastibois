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

# # %% Load data
# zarr_file = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_processed.zarr"
# zfile = zarr.open(zarr_file)

# # %% T, Z, C, X, Y (15, 11, 3, 1024, 1024)
# data = zfile['exp1']

# # %%
# def pretty_plot(image): 
#     p5, p95 = np.percentile(image, [1, 99])
#     plt.imshow(image, cmap='gray', vmin=p5, vmax=p95)
#     plt.show()

# from skimage.color import label2rgb

# def pplot_img_mask(image, mask, figsize=(10, 5)):
#     p5, p95 = np.percentile(image, [1, 99])
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#     ax1.imshow(image, cmap='gray', vmin=p5, vmax=p95)
#     ax2.imshow(label2rgb(mask))
#     ax1.axis('off')
#     ax2.axis('off')
#     plt.show()
    
# # %% Plot some examples
# pretty_plot(data[0, 0, 0])
# pretty_plot(data[0, 0, 1])
# pretty_plot(data[0, 0, 2])
# # %%
# model = models.Cellpose(model_type='cyto2', gpu=True)

# # %%
# membrane_idx = 0
# nucleus_idx = 2
# masks, flows, styles, diams = model.eval(
#     data[0], 
#     anisotropy=10, 
#     diameter=120, 
#     channels=[membrane_idx + 1, nucleus_idx + 1], 
#     channel_axis=1, 
#     do_3D=True
# )
# # %% Show an example
# idx = 7
# pplot_img_mask(data[0, idx, membrane_idx], masks[idx])
# # %%


# #%% create raw and masks group
# zarr_file = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_RawMasks.zarr"

# # with zarr.open(zarr_file,'w') as zarrdir:
# #     zarrdir.create_dataset('raw',data=zfile['exp1'])
# #     print('copied raw, zarrfile shape = ',zarrdir['raw'].shape)
# #     zarrdir.create_dataset('masks',shape=datazarr['raw'][:,:,0,:,:].shape)

# datazarr = zarr.open(zarr_file,'r')

# #making sure theres data in ['masks']
# fig,ax = plt.subplots(1,int(datazarr['masks'].shape[0]),figsize=(50, 10))
# for i in range(int(datazarr['masks'].shape[0])):
#     ax[i].imshow(datazarr['masks'][i,3])
#     ax[i].set_title(f'time {i}')
#     ax[i].set_axis_off()
# plt.show()


#%% load and prepare zarrs for cellpose
# zarr_file = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_RawMasks.zarr"

# # datazarr = zarr.open(zarr_file,'r+')
# datazarr = zarr.open(zarr_file,'r')


# #create new zarr for raw + masks + flow probabilities based on a RawMasks.zarr file
RMP_zarr_path = '/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_RawMasksProbs.zarr'
# with zarr.open(RMP_zarr_path,'w') as RMP_zarr:
#     RMP_zarr = zarr.open(RMP_zarr_path,'w')
#     RMP_zarr.create_dataset('raw',data=datazarr['raw'])
#     RMP_zarr.create_dataset('masks',shape=datazarr['raw'][:,:,0,:,:].shape)
#     RMP_zarr.create_dataset('probs',shape=datazarr['raw'][:,:,0,:,:].shape)
#     print('created RMP_zarr dataset')


# #%% cellpose everything
# model = models.Cellpose(model_type='cyto2', gpu=True)
# membrane_idx = 0
# nucleus_idx = 2

# RMP_zarr = zarr.open(RMP_zarr_path,'r+')

# for i in range(RMP_zarr['raw'].shape[0]): #cellpose whole z-stack every timepoint and save to zarr['masks']
#     img = RMP_zarr['raw'][i]
    
#     masks, flows, styles, diams = model.eval(img, 
#     anisotropy=10, 
#     diameter=120, 
#     channels=[membrane_idx + 1, nucleus_idx + 1], 
#     channel_axis=1, 
#     do_3D=True)

#     RMP_zarr['masks'][i] = masks
#     RMP_zarr['probs'][i] = 1/(1+np.exp(-flows[2])) #apply sigmoid to probabilities

# print('finished masks and probs')







# import napari
# viewer = napari.Viewer()
# viewer.add_image(zarrtest['raw'])

# %% create foreground background


# RMP_zarr = zarr.open(RMP_zarr_path,'r+')
# datanp = RMP_zarr['masks'].astype(int)[:]
# datanp[datanp != 0] = 1
# RMP_zarr.create_dataset('fgbg', data = datanp)
# print('completed')
# print('shape fgbg folder',RMP_zarr['fgbg'].shape)