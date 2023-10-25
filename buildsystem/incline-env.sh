source /etc/profile.d/modules.sh
module purge
module load gcc/8.4.0
module load rocm/5.3.0
source ./buildsystem/spack/incline/modules/dependencies.sh

