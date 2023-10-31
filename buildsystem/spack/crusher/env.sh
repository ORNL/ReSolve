#!/bin/bash
#
module purge
# Load system python
module load cray-python/3.10.10
# Load tcl modules that configure env vars
module load git/2.31.1
module load craype-accel-amd-gfx90a

# Define environment variables for where spack stores key files
# For now, SPACK_INSTALL is the path where everything spack related is installed
# If you want to modify the module install path, edit the spack.yaml manually
BASE=/lustre/orion/csc359/proj-shared/resolve/spack-ci-crusher

export SPACK_INSTALL=/lustre/orion/csc359/proj-shared/resolve/spack-ci-crusher
export SPACK_MODULES=modules
export SPACK_CACHE=$BASE/spack-cache
export SPACK_MIRROR=$BASE/spack-mirror
export SPACK_PYTHON=$CRAY_PYTHON_PREFIX
export SPACK_DISABLE_LOCAL_CONFIG=1
