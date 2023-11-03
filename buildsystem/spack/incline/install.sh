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
export MY_CLUSTER=incline
. buildsystem/load-spack.sh &&
spack develop --no-clone --path=$(pwd) resolve@develop &&
spack concretize -f &&
spack install -j 64 llvm-amdgpu &&
spack load llvm-amdgpu &&
./buildsystem/configure-modules.sh 64

