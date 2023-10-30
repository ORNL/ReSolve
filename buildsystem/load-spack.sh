#!/bin/bash

# Enforce running from resolve root dir
if [[ ! -f $PWD/buildsystem/build.sh ]]; then
  echo 'Please run this script from the top-level resolve source directory.'
  exit 1
fi

# Need to know which platform we are configuring for
MY_CLUSTER="${MY_CLUSTER:?MY_CLUSTER is unset. Please set manually.}"
[[ -z $MY_CLUSTER ]] && return 1 

# There must be an existing folder for the cluster
if [ ! -d "./buildsystem/spack/${MY_CLUSTER}" ]
then
  echo "${MY_CLUSTER} did not match any directories in /buildsystem/spack/"
  echo "Try one of the following platforms: "
  echo $(ls -d ./buildsystem/spack/*/ | tr '\n' '\0' | xargs -0 -n 1 basename | grep -v spack )
  return
fi

base="./buildsystem/spack/${MY_CLUSTER}"

# There needs to be an existing spack.yaml
if [ ! -f "$base/spack.yaml" ]
then
  echo "No spack.yaml in $base/spack.yaml. Add this and try again."
  return
fi

# We must configure with platform specific environment before loading spack
if [ ! -f "$base/env.sh" ]
then
  echo "No environment (env.sh) script detected in $base."
  echo "Please add a script that creates a compatible shell environment."
  echo "You must set the following variables that affect your spack configuration."
  echo "TODO - add updated description of all spack platform variables" # TODO
  return
fi

# Source base platform environment
source $base/env.sh &&

# Make sure SPACK_INSTALL is set, so we aren't installing somewhere undesired
SPACK_INSTALL="${SPACK_INSTALL:?SPACK_INSTALL is unset. $base/env.sh should be edited to configure this}" &&

# Load spack
source ./buildsystem/spack/spack/share/spack/setup-env.sh &&
# Create directory for environment
SPACKENV=$(pwd)/spack-env-$MY_CLUSTER &&
(rm -rfd $SPACKENV || true) && 

# Use a directory based environment, and decorate command line
spack env create -d $SPACKENV &&

# Use git version of config
cp $base/spack.yaml $SPACKENV &&
spack env activate -p $SPACKENV &&

# Print relevant spack config for sanity check of environment.
echo "spack configuration will be installed into $SPACK_INSTALL" &&
mkdir -p $SPACK_INSTALL &&
mkdir -p $SPACK_CACHE &&

# Print config if configured successfully
if [ $? -eq 0 ] && [ "$1" = "-v" ]; then
  spack config get config
fi
