#!/bin/bash

# Load system python
module load python/3.9-anaconda3

# Load tcl modules that configure env vars
module load gcc/10.2
module load git/2.31.1

# Define environment variables for where spack stores key files
# For now, SPACK_INSTALL is the path where everything spack related is installed
# If you want to modify the module install path, edit the spack.yaml manually
BASE=/gpfs/wolf/proj-shared/csc359/resolve/spack-ci
export SPACK_INSTALL=$BASE/install
export SPACK_CACHE=$BASE/../$(whoami)/spack-cache
export SPACK_MIRROR=/gpfs/wolf/csc359/world-shared/spack-ci/mirror
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_USER_CACHE_PATH=$BASE/../$(whoami)
export SPACK_PYTHON=$(which python)

export tempdir=$SPACK_CACHE
export TMP=$SPACK_CACHE
export TMPDIR=$SPACK_CACHE

