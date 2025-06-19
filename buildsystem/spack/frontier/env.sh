#!/bin/bash
#
module reset
# Load system python
module load cray-python
# Load tcl modules that configure env vars
module load git
module load craype-accel-amd-gfx90a

# Define environment variables for where spack stores key files
# For now, SPACK_INSTALL is the path where everything spack related is installed
# If you want to modify the module install path, edit the spack.yaml manually
BASE=/lustre/orion/stf006/world-shared/nkouk/resolve/spack-install

export SPACK_INSTALL=/lustre/orion/stf006/world-shared/nkouk/resolve/spack-install
export SPACK_MODULES=modules
export SPACK_CACHE=$BASE/spack-cache
export SPACK_MIRROR=$BASE/spack-mirror
export SPACK_PYTHON=$CRAY_PYTHON_PREFIX
export SPACK_DISABLE_LOCAL_CONFIG=1
