"""
This script demonstrates creating a Cherab ToroidalVoxelGrid from a list
of grid cells. It also calculates the grid extras, most importantly the
grid's laplacian operator which is used as a regularisation operator in
inversions.

The resulting grid is saved in the
cherab.mastu.bolometry.grid_construction directory. This makes the grid
available to use with the
cherab.mastu.bolometry.load_standard_voxel_grid function. This script
must be run before the load_standard_voxel_grid function is called, in
order to generate the necessary save file.
"""
import pickle
import os
import numpy as np

import cherab.mastu.bolometry.grid_construction
from cherab.mastu.bolometry.grid_construction.wall_masks import CORE_POLYGON_MASK
from cherab.mastu.bolometry.grid_construction.core_rectilinear_reference_grid import CORE_RECTILINEAR_REFERENCE_GRID


nx = 29
ny = 69

enclosed_cells = []
grid_mask = np.empty((nx, ny), dtype=bool)
grid_index_2D_to_1D_map = {}
grid_index_1D_to_2D_map = {}


# Identify the cells that are enclosed by the polygon,
# simultaneously write out grid mask and grid map.
unwrapped_cell_index = 0
for ix in range(nx):
    for iy in range(ny):
        _, p1, p2, p3, p4 = CORE_RECTILINEAR_REFERENCE_GRID[ix][iy]

        # if any points are inside the polygon, retain this cell
        if (CORE_POLYGON_MASK(p1.x, p1.y) or CORE_POLYGON_MASK(p2.x, p2.y)
            or CORE_POLYGON_MASK(p3.x, p3.y) or CORE_POLYGON_MASK(p4.x, p4.y)):
            grid_mask[ix, iy] = True
            grid_index_2D_to_1D_map[(ix, iy)] = unwrapped_cell_index
            grid_index_1D_to_2D_map[unwrapped_cell_index] = (ix, iy)

            enclosed_cells.append((p1, p2, p3, p4))
            unwrapped_cell_index += 1
        else:
            grid_mask[ix, iy] = False


num_cells = len(enclosed_cells)


grid_data = np.empty((num_cells, 4, 2))  # (number of cells, 4 coordinates, x and y values = 2)
for i, row in enumerate(enclosed_cells):
    p1, p2, p3, p4 = row
    grid_data[i, 0, :] = p1.x, p1.y
    grid_data[i, 1, :] = p2.x, p2.y
    grid_data[i, 2, :] = p3.x, p3.y
    grid_data[i, 3, :] = p4.x, p4.y


# Try making grid laplacian matrix for spatial regularisation
grid_laplacian = np.zeros((num_cells, num_cells))

for ith_cell in range(num_cells):

    # get the 2D mesh coordinates of this cell
    ix, iy = grid_index_1D_to_2D_map[ith_cell]

    neighbours = 0

    try:
        n1 = grid_index_2D_to_1D_map[ix-1, iy]  # neighbour 1
        grid_laplacian[ith_cell, n1] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n2 = grid_index_2D_to_1D_map[ix-1, iy+1]  # neighbour 2
        grid_laplacian[ith_cell, n2] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n3 = grid_index_2D_to_1D_map[ix, iy+1]  # neighbour 3
        grid_laplacian[ith_cell, n3] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n4 = grid_index_2D_to_1D_map[ix+1, iy+1]  # neighbour 4
        grid_laplacian[ith_cell, n4] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n5 = grid_index_2D_to_1D_map[ix+1, iy]  # neighbour 5
        grid_laplacian[ith_cell, n5] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n6 = grid_index_2D_to_1D_map[ix+1, iy-1]  # neighbour 6
        grid_laplacian[ith_cell, n6] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n7 = grid_index_2D_to_1D_map[ix, iy-1]  # neighbour 7
        grid_laplacian[ith_cell, n7] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n8 = grid_index_2D_to_1D_map[ix-1, iy-1]  # neighbour 8
        grid_laplacian[ith_cell, n8] = -1
        neighbours += 1
    except KeyError:
        pass

    grid_laplacian[ith_cell, ith_cell] = neighbours


grid = {
    'voxels': grid_data,
    'index_2D_to_1D_map': grid_index_2D_to_1D_map,
    'index_1D_to_2D_map': grid_index_1D_to_2D_map,
    'mask': grid_mask,
    'laplacian': grid_laplacian,
}

# Save the files in the same directory as the loader module
directory = os.path.split(cherab.mastu.bolometry.grid_construction.__file__)[0]
file_name = "core_rectilinear_grid.pickle"
file_path = os.path.join(directory, file_name)
with open(file_path, "wb") as f:
    pickle.dump(grid, f)


print("cells found = {}".format(num_cells))
