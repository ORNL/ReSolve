#!/bin/bash
#BSUB -P CSC359
#BSUB -W 1:30
#BSUB -nnodes 1
#BSUB -c 42
#BSUB -J resolve_spack
#BSUB -alloc_flags maximizegpfs
#BSUB -o spack_install.%J.output
#BSUB -e spack_install.%J.output

export MY_CLUSTER=ascent
. buildsystem/load-spack.sh &&
spack mirror add local file://$SPACK_MIRROR &&
# This is necessary?
spack mirror add spack-public file://$SPACK_MIRROR &&
spack mirror list &&
spack develop --no-clone --path=$(pwd) resolve@develop &&
jsrun -n 1 -c 40 ./buildsystem/configure-modules.sh 40

