#!/bin/bash

# Enforce running from resolve root dir
if [[ ! -f $PWD/buildsystem/build.sh ]]; then
  echo 'Please run this script from the top-level resolve source directory.'
  exit 1
fi

SPACK_INSTALL="${SPACK_INSTALL:?SPACK_INSTALL is unset. Make sure to source load_spack.sh first}"

# Assuming MY_CLUSTER is configured
# TODO - fix this to be POSIX standardized with tolower across all scripts
base="./buildsystem/spack/${MY_CLUSTER}"

spack install -j $1 && \
spack module tcl refresh -y && \
	
# We will create a new modules file, with the first line being the module path
mkdir -p $base/modules && \
# Note we redirect and destroy old file
arch=$(spack arch) && \
echo Arch for module path being used is $arch && \
echo module use -a $SPACK_INSTALL/modules/$arch &> $base/modules/dependencies.sh && \
spack module tcl loads -r -x resolve resolve &>> $base/modules/dependencies.sh

# This makes the module and installation location globally readable which isn't ideal.
# Sticking to this avoids permission issues for other group members, but you
# should make sure that the install location is in a folder that is only
# accessible to group members
# Since we use this in CI and other users will create files we cannot chmod,
# We need to allow this command to fail
chmod -R ugo+wrx $SPACK_INSTALL/modules > /dev/null 2>&1
