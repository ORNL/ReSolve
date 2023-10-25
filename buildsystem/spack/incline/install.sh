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

export MY_CLUSTER=incline
. buildsystem/load-spack.sh &&
spack develop --no-clone --path=$(pwd) resolve@develop &&
./buildsystem/configure-modules.sh 64

