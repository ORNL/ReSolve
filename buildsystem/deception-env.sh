source /etc/profile.d/modules.sh

module purge
module load cmake/3.26.0
module load gcc/10.2.0
module load cuda/11.4

# if you want to use exago's suite-sparse uncomment the below comments.
module use -a /qfs/projects/exasgd/src/ci-deception/ci-modules/linux-centos7-zen2
module load suite-sparse-5.10.1-gcc-10.2.0-add65sb
