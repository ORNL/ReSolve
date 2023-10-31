#!/bin/bash
#SBATCH -A csc359_crusher
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -J resolve_spack
#SBATCH -o spack_install.%J.output
#SBATCH -e spack_install.%J.output
#SBATCH -t 240

# Configure https proxy because spack is going to do some things with git
export all_proxy="socks://proxy.ccs.ornl.gov:3128"
export ftp_proxy="ftp://proxy.ccs.ornl.gov:3128"
export http_proxy="http://proxy.ccs.ornl.gov:3128"
export https_proxy="http://proxy.ccs.ornl.gov:3128"
export HTTP_PROXY="http://proxy.ccs.ornl.gov:3128"
export HTTPS_PROXY="http://proxy.ccs.ornl.gov:3128"
export proxy="proxy.ccs.ornl.gov:3128"
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov,*.olcf.ornl.gov,*.ncrc.gov'

export MY_CLUSTER=crusher
. buildsystem/load-spack.sh &&
spack mirror add local file://$SPACK_MIRROR &&
# This is necessary?
spack mirror add spack-public file://$SPACK_MIRROR &&
spack mirror list &&
spack develop --no-clone --path=$(pwd) resolve@develop &&
srun -n 1  ./buildsystem/configure-modules.sh 40
