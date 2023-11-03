#!/bin/bash

source /etc/profile.d/modules.sh
module purge

# Load system python and gcc
module load python/miniconda4.12
source /share/apps/python/miniconda4.12/etc/profile.d/conda.sh
module load gcc/8.4.0

# Define environment variables for where spack stores key files
# For now, SPACK_INSTALL is the path where everything spack related is installed
# If you want to modify the module install path, edit the spack.yaml manually
BASE=/qfs/projects/exasgd/resolve/spack-ci
export SPACK_INSTALL=$BASE/install
export SPACK_MIRROR=$BASE/../$(whoami)/spack-mirror
export SPACK_CACHE=$BASE/../$(whoami)/spack-cache
export SPACK_DISABLE_LOCAL_CONFIG=1
export SPACK_PYTHON=$(which python3)

export tempdir=$SPACK_CACHE
export TMP=$SPACK_CACHE
export TMPDIR=$SPACK_CACHE

