#!/bin/bash

exit() {
  # Clear all trap handlers so this isn't echo'ed multiple times, potentially
  # throwing off the CI script watching for this output
  trap - `seq 1 31`

  # If called without an argument, assume not an error
  local ec=${1:-0}

  # Echo the snippet the CI script is looking for
  echo BUILD_STATUS:${ec}

  # Actually exit with that code, although it won't matter in most cases, as CI
  # is only looking for the string 'BUILD_STATUS:N'
  builtin exit ${ec}
}

# This will be the catch-all trap handler after arguments are parsed.
cleanup() {
  # Clear all trap handlers
  trap - `seq 1 31`

  # When 'trap' is invoked, each signal handler will be a curried version of
  # this function which has the first argument bound to the signal it's catching
  local sig=$1

  echo
  echo Exit code $2 caught in build script triggered by signal ${sig}.
  echo

  exit $2
}

if [[ ! -f $PWD/buildsystem/build.sh ]]; then
  echo 'Please run this script from the top-level Resolve source directory.'
  exit 1
fi

export SRCDIR=${SRCDIR:-$PWD}
export BUILDDIR=${BUILDDIR:-$PWD/build}
export INSTALLDIR=${INSTALLDIR:-$PWD/install}
export MAKEARGS=${MAKEARGS:-"-j"}
export CTESTARGS=${CTESTARGS:-"--output-on-failure"}
export BUILD_VERBOSE=0
export BUILD=${BUILD:-1}
export TEST=${TEST:-1}

usage() {
  echo "Usage: ./buildsystem/build.sh [options]

--------------------------------------------------------------------------------

Long Description:

  This script is the entry point for Resolvess continuous integration and default
  build configuration. 

Clusters:

  Each cluster has tcl modules that need to be loaded before invoking CMake. These
  are configured through an environment variable \"MY_CLUSTER\". Current available
  clusters for configuration are:

    - deception
    - ascent 
    - incline

  Run \`export MY_CLUSTER=deception\` or invoke the build script with:

    \`MY_CLUSTER=ascent ./buildsystem/build.sh\`

--------------------------------------------------------------------------------

Options:

  --build-only      Only run the build stage of the script. This is useful for
                    local development.

  --test-only       Only run the test stage of the script. This should be ran
                    before every push to the repository or pull/merge request.
                    This run takes a significant amound of time. If you omit
                    the --*-only options and just run a particular job, tests
                    will also be ran.

  --verbose        Print all executed commands to the terminal. This is useful 
                   for debugging, but it will be disabled in CI by default to 
                   prevent hitting the job log limit.

--------------------------------------------------------------------------------

See Resolve's latest developer guidelines for more information on developing
Resolve: https://code.ornl.gov/ecpcitest/exasgd/resolve/-/blob/develop/CONTRIBUTING.md

--------------------------------------------------------------------------------

"
}

while [[ $# -gt 0 ]]; do
  case $1 in
  --build-only)
    export TEST=0 BUILD=1
    shift
    ;;
  --test-only)
    export TEST=1 BUILD=0
    shift
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  --verbose|-v)
    export BUILD_VERBOSE=1
    shift
    ;;
  *)
    echo "Argument $1 not recognized."
    usage
    exit 1
    ;;
  esac
done

# Trap all signals and pass signal to the handler so we know what signal was
# sent in CI
for sig in `seq 1 31`; do
  trap "cleanup $sig \$? \$LINENO" $sig
done

if [ $BUILD_VERBOSE == 1 ]
then
  # verbose mode: print out all shell functions
  set -xv
fi

module purge

if [[ ! -v MY_CLUSTER ]]
then
  export MY_CLUSTER=`uname -n | sed -e 's/[0-9]//g' -e 's/\..*//'`
fi

# Correctly identify clusters based on hostname
case $MY_CLUSTER in
  incline*|dmi*)
    export MY_CLUSTER=incline
    ;;
  dl*|deception|*fat*)
    export MY_CLUSTER=deception
    ;;
  ascent*)
    export MY_CLUSTER=ascent
    ;;
  *)
    echo "Cluster $MY_CLUSTER not identified - you'll have to set relevant variables manually."
    exit 1
    ;;
esac

varfile="${SRCDIR}/buildsystem/${MY_CLUSTER}-env.sh"

if [[ ! -v MY_CLUSTER ]]
then
  echo "MY_CLUSTER" unset && exit 1
fi

if [[ -f "$varfile" ]]; then
  source $varfile || { echo "Could not source $varfile"; exit 1; }
else
  echo "No cluster variable file configured for ${MY_CLUSTER}. Try one of:\n"
  echo "deception, incline, ascent." && exit 1
fi

echo "Paths:"
echo "Source dir: $SRCDIR"
echo "Build dir: $BUILDDIR"
echo "Install dir: $INSTALLDIR"
echo "Path to buildsystem script: $SRCDIR/buildsystem/build.sh"

module list

if [[ $BUILD -eq 1 ]]; then
  [ -d $BUILDDIR ] && rm -rf $BUILDDIR
  mkdir -p $BUILDDIR

  [ -d $INSTALLDIR ] && rm -rf $INSTALLDIR
  mkdir -p $INSTALLDIR

  pushd $BUILDDIR

  echo
  echo Configuring
  echo
  eval "cmake -S .. --preset ${MY_CLUSTER}" || exit 1

  echo
  echo Building
  echo
  make $MAKEARGS || exit 1

  echo
  echo Installing
  echo
  make install || exit 1
  popd
fi

if [[ $TEST -eq 1 ]]; then
  pushd $BUILDDIR
  echo
  echo Testing
  echo
  ctest $CTESTARGS || exit 1
  make test_install || exit 1
  popd
fi

if [ $BUILD_VERBOSE == 1 ]
then
  set +xv
fi

exit 0
