"""
This script is used to combine the results of separate batch jobs to
calculate the sensitivity matrices of the bolometers. Calculations
may be run as separate jobs for sub-sets of grid cells to reduce memory
usage or access more cores than available on a single compute node.
"""
from collections import defaultdict
import os
import sys
import numpy as np


def combine_cells(*file_names):
    """
    Read the entire sensitivity matrix for all grid cells.

    Given a series of file names, containing the output of Cherab
    sensitivity calculations for subsets of grid cells, this function
    collects the outputs and combines them into a single output file for
    the camera.

    The files in *file_names are assumed to be numpy save files (with
    extension npy), which contain the sensitivity matrix elements for a
    subset of grid cells for a given channel.

    The filename format of the files is expected to be:
    <grid>_<camera>_bolo_<istart>_<iend>.npy
    This naming convention is used to infer the array elements of the
    sensitivity matrices to be extracted.

    The output file containing the calculations for all cells for all
    channels will then be named:
    <grid>_<camera>_bolo.npy.

    This function returns nothing, but results in the output file
    being written to the current working directory.
    """
    # Ensure all files refer to the same grid and camera
    grid, camera = file_names[0].split("_")[:2]
    if not all(f.split("_")[0] == grid for f in file_names):
        raise ValueError("All file names must have the same grid name.")
    if not all(f.split("_")[1] == camera for f in file_names):
        raise ValueError("All file names must have the same camera name.")

    # Get the number of channels by reading the first file
    nchan = np.load(file_names[0]).shape[0]
    ncells = 0
    foil_data = defaultdict(list)
    for save_file in file_names:
        filename = os.path.basename(save_file)
        name, ext = filename.split(".")
        if ext != "npy":
            raise ValueError("All files should be in npy Numpy save file format")
        try:
            istart, iend = name.split("_")[-2:]
            istart = int(istart)
            iend = int(iend)
        except ValueError:
            raise ValueError(
                "File name should be of the format "
                "<grid>_<camera>_bolo_<istart>_<iend>.npy"
            )
        if iend > ncells:
            ncells = iend
        data = np.load(save_file)
        foil_data[(istart, iend)] = data
    sensitivities = np.empty((nchan, ncells))
    for (istart, iend), cell_data in foil_data.items():
        sensitivities[:, istart:iend] = cell_data

    # Write the complete camera sensitivity data to file
    output_file_name = "{}_{}_bolo.npy".format(grid, camera)
    print("Writing {}".format(output_file_name))
    np.save(output_file_name, sensitivities)


def main():
    """Function called when script is run."""
    if len(sys.argv) < 3:
        raise ValueError("Usage: {} [list of pickle files to combine]"
                         .format(sys.argv[0]))
    combine_cells(*sys.argv[1:])


if __name__ == "__main__":
    main()
