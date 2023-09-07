# %%
import numpy as np
import zarr
import gunpowder as gp
import matplotlib.pyplot as plt

# %%


# for testing, get another dataset


zarr_path = (
    "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/glass_60x_023_RawMasksProbs.zarr"
)
zarrgp_path = "/mnt/efs/shared_data/YeastiBois/zarr_files/Tien/gunpowder.zarr"

log = gp.ArrayKey("log")

raw = gp.ArrayKey("RAW")
gt = gp.ArrayKey("gt")

predictions = gp.ArrayKey("predictions")
affinities = gp.ArrayKey("affinities")
lsds = gp.ArrayKey("lsds")

# #for a single timepoint
# source = gp.ZarrSource(
#     zarrgp_path,  # the zarr container
#     {raw: 'raw/t0', gt:'fgbg/t0'},  # dict with all zarrs mapped to my array keys from above
#     {raw: gp.ArraySpec(voxel_size=(10,1,1),interpolatable=True),
#      gt: gp.ArraySpec(voxel_size=(10,1,1),interpolatable=False)}  # meta-information
# )

zfile = zarr.open(zarrgp_path, "r")
[key for key in zfile["raw"]]

# multiple timepoints, groupname = 'raw'/folders for datasets called f't{i}'
source = tuple(
    gp.ZarrSource(
        zarrgp_path,  # the zarr container
        {
            raw: f"raw/{key}",
            gt: f"masks/{key}",
        },  # dict with all zarrs mapped to my array keys from above
        {
            raw: gp.ArraySpec(voxel_size=(10, 1, 1), interpolatable=True),
            gt: gp.ArraySpec(voxel_size=(10, 1, 1), interpolatable=False),
        },  # meta-information
    )
    for key in zfile["raw"]
)

# %%
request = gp.BatchRequest()
voxel_size = gp.Coordinate((10, 1, 1))
shape = gp.Coordinate((1, 64, 64))
roi = gp.Roi((0, 0, 0), shape * voxel_size)
request[raw] = roi
request[gt] = roi

request[log] = gp.ArraySpec(nonspatial=True)

pipeline = (
    source
    + gp.RandomProvider(random_provider_key=log)
    + gp.Normalize(raw)
    + gp.SimpleAugment()
    + gp.IntensityAugment(raw)
    + gp.NoiseAugment(raw)
    + gp.AddAffinities(
        affinity_neighborhood=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        labels=gt,
        affinities=affinities,
    )
)


# %%
with gp.build(pipeline):
    batch = pipeline.request_batch(request)

print(batch[raw].data.shape)
print(batch[log].data)

# %%
plt.imshow(np.transpose(batch[raw].data.squeeze(), (1, 2, 0)))
# %%
plt.imshow(batch[gt].data.squeeze())
# %%
