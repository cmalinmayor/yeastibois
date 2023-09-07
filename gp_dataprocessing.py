#%%
import zarr
import numpy as np


zarr_path = '/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_RawMasksProbs.zarr'
zarrgp_path = '/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/gunpowder.zarr'
#raw data currently is (time,z,channel,y,x)

originalzarr = zarr.open(zarr_path,'r')
gpzarr = zarr.open(zarrgp_path,'w')
gpzarr.create_group('raw')
gpzarr.create_group('fgbg')
gpzarr.create_group('masks')
for i in range(originalzarr['raw'].shape[0]):
    data = np.array(originalzarr['raw'][i])
    data = np.swapaxes(data,0,1)
    gpzarr['raw'][f't{i}'] = data
    gpzarr['fgbg'][f't{i}'] = originalzarr['fgbg'][i]
    gpzarr['masks'][f't{i}'] = originalzarr['masks'][i]