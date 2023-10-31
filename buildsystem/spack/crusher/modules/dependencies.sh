module use -a /lustre/orion/csc359/proj-shared/resolve/spack-ci-crusher/modules/linux-sles15-zen3
# cmake@=3.23.2%clang@=16.0.0-rocm5.6.0-mixed~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-sles15-zen3
module load cmake/3.23.2-clang-16.0.0-rocm5.6.0-mixed-dav4a3k
# gmake@=4.4.1%clang@=16.0.0-rocm5.6.0-mixed~guile build_system=generic arch=linux-sles15-zen3
module load gmake/4.4.1-clang-16.0.0-rocm5.6.0-mixed-zhuamlj
# hip@=5.6.0%clang@=16.0.0-rocm5.6.0-mixed~cuda+rocm build_system=cmake build_type=Release generator=make patches=aee7249,c2ee21c,e73e91b arch=linux-sles15-zen3
module load hip/5.6.0-clang-16.0.0-rocm5.6.0-mixed-sl6ruyb
# hsa-rocr-dev@=5.6.0%clang@=16.0.0-rocm5.6.0-mixed+image+shared build_system=cmake build_type=Release generator=make patches=9267179 arch=linux-sles15-zen3
module load hsa-rocr-dev/5.6.0-clang-16.0.0-rocm5.6.0-mixed-hhzg72u
# llvm-amdgpu@=5.6.0%clang@=16.0.0-rocm5.6.0-mixed~link_llvm_dylib~llvm_dylib~openmp+rocm-device-libs build_system=cmake build_type=Release generator=ninja patches=a08bbe1,b66529f,d35aec9 arch=linux-sles15-zen3
module load llvm-amdgpu/5.6.0-clang-16.0.0-rocm5.6.0-mixed-rczbatz
# rocblas@=5.6.0%clang@=16.0.0-rocm5.6.0-mixed+tensile amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocblas/5.6.0-clang-16.0.0-rocm5.6.0-mixed-fn6g3hv
# rocsparse@=5.6.0%clang@=16.0.0-rocm5.6.0-mixed~test amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocsparse/5.6.0-clang-16.0.0-rocm5.6.0-mixed-v7vzlm6
# gmp@=6.2.1%clang@=16.0.0-rocm5.6.0-mixed+cxx build_system=autotools libs=shared,static patches=69ad2e2 arch=linux-sles15-zen3
module load gmp/6.2.1-clang-16.0.0-rocm5.6.0-mixed-tjwo3xg
# libiconv@=1.17%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load libiconv/1.17-clang-16.0.0-rocm5.6.0-mixed-d5acvgj
# diffutils@=3.9%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools arch=linux-sles15-zen3
module load diffutils/3.9-clang-16.0.0-rocm5.6.0-mixed-avn5tx2
# libsigsegv@=2.14%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools arch=linux-sles15-zen3
module load libsigsegv/2.14-clang-16.0.0-rocm5.6.0-mixed-asnjual
# m4@=1.4.19%clang@=16.0.0-rocm5.6.0-mixed+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-sles15-zen3
module load m4/1.4.19-clang-16.0.0-rocm5.6.0-mixed-6ft6syx
# metis@=5.1.0%clang@=16.0.0-rocm5.6.0-mixed~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903 arch=linux-sles15-zen3
module load metis/5.1.0-clang-16.0.0-rocm5.6.0-mixed-pbd7yry
# autoconf@=2.69%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools patches=7793209 arch=linux-sles15-zen3
module load autoconf/2.69-clang-16.0.0-rocm5.6.0-mixed-p42v7id
# autoconf-archive@=2023.02.20%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools arch=linux-sles15-zen3
module load autoconf-archive/2023.02.20-clang-16.0.0-rocm5.6.0-mixed-il2o2kz
# automake@=1.16.3%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools arch=linux-sles15-zen3
module load automake/1.16.3-clang-16.0.0-rocm5.6.0-mixed-eroaj55
# libtool@=2.4.6%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools arch=linux-sles15-zen3
module load libtool/2.4.6-clang-16.0.0-rocm5.6.0-mixed-wkjhjyo
# texinfo@=6.5%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools patches=12f6edb,1732115 arch=linux-sles15-zen3
module load texinfo/6.5-clang-16.0.0-rocm5.6.0-mixed-vqdeuda
# mpfr@=4.2.0%clang@=16.0.0-rocm5.6.0-mixed build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load mpfr/4.2.0-clang-16.0.0-rocm5.6.0-mixed-4luvnxj
# openblas@=0.3.17%gcc@=12.2.0-mixed~bignuma~consistent_fpcsr~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-sles15-zen3
module load openblas/0.3.17-gcc-12.2.0-mixed-maehd6o
# suite-sparse@=5.13.0%clang@=16.0.0-rocm5.6.0-mixed~cuda~graphblas~openmp+pic build_system=generic arch=linux-sles15-zen3
module load suite-sparse/5.13.0-clang-16.0.0-rocm5.6.0-mixed-7ozrbmp
# resolve@=develop%clang@=16.0.0-rocm5.6.0-mixed~cuda~ipo+klu+rocm amdgpu_target=gfx90a build_system=cmake build_type=Release dev_path=/ccs/home/dane678/resolve generator=make arch=linux-sles15-zen3
## module load resolve/develop-clang-16.0.0-rocm5.6.0-mixed-wu4veii
