#!/bin/bash
#SBATCH -A exasgd
#SBATCH -p slurm
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -J resolve_spack
#SBATCH -o spack_install.%J.output
#SBATCH -e spack_install.%J.output
#SBTACH -t 240

export MY_CLUSTER=deception
. buildsystem/load-spack.sh &&
spack develop --no-clone --path=$(pwd) resolve@develop &&
./buildsystem/configure-modules.sh 64
