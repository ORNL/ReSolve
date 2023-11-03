#!/bin/bash
#SBATCH -A exasgd
#SBATCH -p incline
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --gres=gpu:1
#SBATCH -J resolve_spack
#SBATCH -o spack_install.%J.output
#SBATCH -e spack_install.%J.output
#SBTACH -t 240

export HTTPS_PROXY=http://proxy01.pnl.gov:3128
export https_proxy=http://proxy01.pnl.gov:3128

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

export MY_CLUSTER=incline
. buildsystem/load-spack.sh &&
spack develop --no-clone --path=$(pwd) resolve@develop &&
spack concretize -f &&
spack install -j 64 llvm-amdgpu &&
spack load llvm-amdgpu &&
./buildsystem/configure-modules.sh 64


EXIT_CODE=$?
# Required to trigger trap handler
exit $EXIT_CODE
