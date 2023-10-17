#!/bin/bash
#BSUB -P CSC359
#BSUB -W 1:30
#BSUB -nnodes 1
#BSUB -c 42
#BSUB -J resolve_spack
#BSUB -alloc_flags maximizegpfs
#BSUB -o spack_install.%J.output
#BSUB -e spack_install.%J.output

# Configure https proxy because spack is going to do some things with git
export all_proxy="socks://proxy.ccs.ornl.gov:3128"
export ftp_proxy="ftp://proxy.ccs.ornl.gov:3128"
export http_proxy="http://proxy.ccs.ornl.gov:3128"
export https_proxy="http://proxy.ccs.ornl.gov:3128"
export HTTP_PROXY="http://proxy.ccs.ornl.gov:3128"
export HTTPS_PROXY="http://proxy.ccs.ornl.gov:3128"
export proxy="proxy.ccs.ornl.gov:3128"
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov,*.olcf.ornl.gov,*.ncrc.gov'

export MY_CLUSTER=ascent
. buildsystem/load-spack.sh &&
spack mirror add local file://$SPACK_MIRROR &&
# This is necessary?
spack mirror add spack-public file://$SPACK_MIRROR &&
spack mirror list &&
spack develop --no-clone --path=$(pwd) resolve@develop &&
jsrun -n 1 -c 40 ./buildsystem/configure-modules.sh 40
