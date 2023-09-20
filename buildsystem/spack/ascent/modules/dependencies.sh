module use -a /gpfs/wolf/proj-shared/csc359/resolve/spack-ci/install/modules/linux-rhel8-power9le
# cmake@=3.22.2%gcc@=10.2~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-rhel8-power9le
module load cmake/3.22.2-gcc-10.2-licrwny
# cuda@=11.4.2%gcc@=10.2~allow-unsupported-compilers~dev build_system=generic arch=linux-rhel8-power9le
module load cuda/11.4.2-gcc-10.2-vlwmnfb
# gnuconfig@=2022-09-17%gcc@=10.2 build_system=generic arch=linux-rhel8-power9le
module load gnuconfig/2022-09-17-gcc-10.2-ujor7tm
# gmake@=4.4.1%gcc@=10.2~guile build_system=autotools arch=linux-rhel8-power9le
module load gmake/4.4.1-gcc-10.2-rnqfmzf
# suite-sparse@=5.10.1%gcc@=10.2~cuda~graphblas~openmp+pic~tbb build_system=generic arch=linux-rhel8-power9le
module load suite-sparse/5.10.1-gcc-10.2-otp7tib
# resolve@=develop%gcc@=10.2+cuda~ipo+klu build_system=cmake build_type=Release cuda_arch=70 dev_path=/gpfs/wolf/proj-shared/csc359/ci/466546 generator=make arch=linux-rhel8-power9le
## module load resolve/develop-gcc-10.2-66cvnuz
