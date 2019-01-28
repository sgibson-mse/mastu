
import pickle
import os

from raysect.core import Point2D
from cherab.tools.inversions import ToroidalVoxelGrid


def load_standard_voxel_grid(system, parent=None, name='', voxel_range=None,
                             primitive_type='csg'):
    """
    Load the standard bolometry reconstruction grid.

    :param system: the name of the bolometer system used
    Currently 'core', 'core_high_res' and 'sxdl' are supported.
    :param parent: the parent node for the grid object.
    :param name: the name of the grid object
    :param voxel_range: if not none, return a subset of the
    grid cells, `voxels[voxel_range]`. Can be either a slice object or
    a range object.
    :param str primitive_type: specifies the representation type of the
    voxels in the grid. 'mesh' and 'csg' are supported. 'mesh' generally
    works well but can have extremely high memory usage on higher
    resolution grids, whereas 'csg' has much lower memory usage but can
    be slower on grids with many non-rectilinear voxels. Both types
    support voxels with arbitrary polygon cross sections.

    Before this function is called, the corresponding pickle file for the
    grid must have already been generated. This can be done by running the
    `/examples/bholometry/grid_generation/make_<system>_grid.py` scripts.
    """

    directory = os.path.split(__file__)[0]

    grid_file = os.path.join(
        directory,
        '{}_rectilinear_grid.pickle'.format(system)
    )

    with open(grid_file, 'rb') as f:
        grid_data = pickle.load(f)
    grid_voxels = grid_data['voxels']
    if voxel_range is not None:
        grid_voxels = grid_voxels[voxel_range]
    voxel_list = [[Point2D(*vertex) for vertex in voxel] for voxel in grid_voxels]
    grid = ToroidalVoxelGrid(voxel_list, parent=parent, name=name,
                             primitive_type=primitive_type)
    return grid


def load_grid_extras(system):
    """
    Load extra information about the reconstruction grid.

    :param system: the name of the grid.
    Currently 'core', 'core_high_res' and 'sxdl' are supported.

    The grid extra information at the moment includes a mapping of the
    1D cell index to 2D grid of cells, and the other way around, as well
    as the isotropic grid laplacian operator for inversion
    regularisation.
    """
    extras_file = os.path.join(
        os.path.split(__file__)[0],
        '{}_rectilinear_grid.pickle'.format(system)
        )
    with open(extras_file, 'rb') as extras_file:
        saved_extras = pickle.load(extras_file)
    # Extras is everything other than the voxel list
    saved_extras.pop('voxels')
    return saved_extras
