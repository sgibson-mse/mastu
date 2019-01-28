#!/bin/bash

# Calculate the sensitivity matrices for a given channel for the high res
# grid, by splitting the grid into a series of sub-lists of cells and
# calculating each sub list in a separate batch job
#
# For lower resolution grids, the calculation is sufficiently short to
# run just a single job, in which case the for loop is unnecessary.
# Instead, just run the single qsub command, leaving out the istart and
# iend varaibles.
#
# This script should be run on Freia, with the directory in which this
# script and calculate_sensitivities.py reside as the current working
# directory.

NSLOTS=10
NJOBS=10

usage() {
    echo "Usage: $0 <camera>"
    echo "<camera> is 'Poloidal' or 'Tangential'"
    exit 1
}

[[ "$#" -eq 1 ]] || usage
camera=$1
[[ "$camera" = "Poloidal" ]] || [[ "$camera" = "Tangential" ]] || usage
camera="${camera}HighRes"

ncells=$(python3 <<EOF
from cherab.mastu.bolometry import load_standard_voxel_grid
grid = load_standard_voxel_grid("core_high_res")
print(len(grid))
EOF
      )
echo "Grid has $ncells cells"
cells_per_job=$((ncells / NJOBS + 1))

for istart in $(seq 0 "$cells_per_job" "$ncells")
do
    iend=$((istart + cells_per_job))
    jobname="${camera}_${istart}_${iend}"
    output="$(pwd)/${jobname}_stdout.log"
    error="$(pwd)/${jobname}_stderr.log"
    qsub <<EOF
#!/bin/bash
#$ -N $jobname
#$ -cwd
#$ -pe smp $NSLOTS
#$ -o $output
#$ -e $error
#$ -m ea
#$ -M $USER
module purge
module load standard python/3.5 uda uda-mast
python3 calculate_sensitivities.py $camera $istart $iend
EOF
done
