module purge
module load rocm/5.6.0
module load PrgEnv-gnu-amd
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load amd-mixed/5.6.0
module load gcc-mixed/12.2.0
module load cray-mpich/8.1.25
module load libfabric
source ./buildsystem/spack/crusher/modules/dependencies.sh
