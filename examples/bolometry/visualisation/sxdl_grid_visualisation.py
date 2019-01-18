"""
This script demonstrates loading the reconstruction grid for the lower
divertor bolometers, and plotting it alongside the MAST-U passive
structures.
"""
import matplotlib.pyplot as plt
import pyuda
from cherab.mastu.bolometry import load_standard_voxel_grid


grid = load_standard_voxel_grid('sxdl')
client = pyuda.Client()
passive_structures = client.geometry('/passive/efit', 50000)

fig, ax = plt.subplots()
grid.plot(ax=ax)
passive_structures.plot(ax_2d=ax, color='k')
ax.set_ylim(top=-1.5)

# Reduce the line width of the grid edges to make it clearer
for patch_collection in ax.collections:
    patch_collection.set_linewidth(0.7)

plt.show()
