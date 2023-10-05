#!/bin/bash

if [[ ! -f $PWD/buildsystem/build.sh ]]; then
  echo 'Please run this script from the top-level Resolve source directory.'
  exit 1
fi

export SRCDIR=${SRCDIR:-$PWD}
export BUILDDIR=${BUILDDIR:-$PWD/build}
export INSTALLDIR=${INSTALLDIR:-$PWD/install}
export BUILD_VERBOSE=0

echo "Paths:"
echo "Source dir: $SRCDIR"
echo "Build dir: $BUILDDIR"
echo "Install dir: $INSTALLDIR"
echo "Path to buildsystem script: $SRCDIR/buildsystem/build.sh"
cd $SRCDIR

usage() {
  echo "Usage: ./buildsystem/build.sh [options]

--------------------------------------------------------------------------------

Long Description:

  This script is the entry point for Resolvess continuous integration and default
  build configuration. 

Clusters:

  By default, this script will run on Ascent. 

--------------------------------------------------------------------------------

Options:

  Not Applicable at the moment.
--------------------------------------------------------------------------------

See Resolve's latest developer guidelines for more information on developing
Resolve: https://code.ornl.gov/ecpcitest/exasgd/resolve/-/blob/develop/CONTRIBUTING.md

--------------------------------------------------------------------------------

"
}

if [ $BUILD_VERBOSE == 1 ]
then
  # verbose mode: print out all shell functions
  set -xv
else
  # don't print out all shell functions 
  set +x
fi

module purge

varfile="$SRCDIR/buildsystem/ascent-env.sh"

if [[ -f "$varfile" ]]; then
  source "$varfile"
  echo Sourced system-specific variables for ascent
fi

module list

mkdir -p build

rm -rf build/*

# Utlizes CMakePresets.json file to set cmake variables such as CMAKE_INSTALL_PREFIX
# preset cluster = CMAKE_INSTALL_PREFIX = /install
cmake -B build --preset cluster && \

cmake --build build

exit $?


