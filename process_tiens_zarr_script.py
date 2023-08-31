#%% import
import zarr
import numpy as np
from process_zebrafish_zarr import process_bette_zarr
import napari

#%% list of available zarr files
zarrdir = '../convertedzarr/'
zarrlist = []
for file in os.listdir(zarrdir):
    zarrlist.append(file)
zarrlist = sorted(zarrlist)
print('zarrlist= ',zarrlist)


#%% process
zarr_to_process_path = os.path.join(zarrdir,zarrlist[3])
process_bette_zarr(zarr_to_process_path)

#%% open zarr in napari
img = os.path.join('../convertedzarr/',zarrlist[4])
zarrimg = zarr.open(img,'r')
viewer = napari.Viewer()
viewer.add_image(zarrimg['exp1'])

#%%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()