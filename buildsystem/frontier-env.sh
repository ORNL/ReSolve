module reset
module load PrgEnv-amd
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load amd/6.3.1
module load rocm/6.3.1
module load cray-mpich
module load libfabric
source ./buildsystem/spack/frontier/modules/dependencies.sh
