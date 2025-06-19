#!/bin/bash

# Load system rocm
source /etc/profile.d/modules.sh
module purge
module load gcc/8.4.0

# These are necessary in order to see GPUs with sbatch
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

# Load spack generated modules
source ./buildsystem/spack/incline/modules/dependencies.sh
