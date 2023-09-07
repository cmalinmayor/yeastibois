#%%
import numpy as np
import matplotlib.pyplot as plt
import zarr

#%%

# Generate a dummy 4D data
zarrpath = '/home/tienc/Documents/trackvids/tracked.zarr/'
zarrfile = zarr.open(zarrpath)
img = zarrfile['img']
label = zarrfile['label']

data = img  # Replace with your data

# Example time point
time_point = 0  # You can loop through time points if needed

# Extract 3D data for the given time point
img3D = data[time_point]

# Create mask for intensity-based filtering, e.g., if you only want to plot intensities above a threshold
# This helps in removing background noise
threshold = 1000
mask = img3D > threshold

# Get coordinates and intensities
z, y, x = np.where(mask)
intensities = img3D[mask]

# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=intensities, cmap='gray')

# Optionally add a colorbar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)

plt.show()