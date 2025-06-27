module use -a /lustre/orion/stf006/world-shared/nkouk/resolve/spack-install/modules/linux-sles15-zen3
# cmake@=3.30.5%rocmcc@=6.3.1~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release patches=dbc3892 arch=linux-sles15-zen3
module load cmake/3.30.5-rocmcc-6.3.1-hv4mzvr
# glibc@=2.38%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load glibc/2.38-rocmcc-6.3.1-3hkbcwm
# gmake@=4.4.1%rocmcc@=6.3.1~guile build_system=generic arch=linux-sles15-zen3
module load gmake/4.4.1-rocmcc-6.3.1-pca4j24
# hip@=6.3.1%rocmcc@=6.3.1~asan~cuda+rocm build_system=cmake build_type=Release generator=make patches=1f65dfe arch=linux-sles15-zen3
module load hip/6.3.1-rocmcc-6.3.1-ar4rzg2
# hsa-rocr-dev@=6.3.1%rocmcc@=6.3.1~asan+image+shared build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load hsa-rocr-dev/6.3.1-rocmcc-6.3.1-wwlghld
# llvm-amdgpu@=6.3.1%rocmcc@=6.3.1~link_llvm_dylib~llvm_dylib+rocm-device-libs build_system=cmake build_type=Release generator=ninja patches=b4774ca arch=linux-sles15-zen3
module load llvm-amdgpu/6.3.1-rocmcc-6.3.1-fuu6pxm
# rocblas@=6.3.1%rocmcc@=6.3.1~asan+tensile amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocblas/6.3.1-rocmcc-6.3.1-m3viuwr
# rocsolver@=6.3.1%rocmcc@=6.3.1~asan+optimal amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocsolver/6.3.1-rocmcc-6.3.1-qfyv2cr
# rocsparse@=6.3.1%rocmcc@=6.3.1~asan~test amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocsparse/6.3.1-rocmcc-6.3.1-y3afx76
# autoconf@=2.72%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load autoconf/2.72-rocmcc-6.3.1-aq7yj4t
# berkeley-db@=18.1.40%rocmcc@=6.3.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-sles15-zen3
module load berkeley-db/18.1.40-rocmcc-6.3.1-6epjpim
# libiconv@=1.17%rocmcc@=6.3.1 build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load libiconv/1.17-rocmcc-6.3.1-pyw3ira
# diffutils@=3.10%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load diffutils/3.10-rocmcc-6.3.1-ltpkf77
# bzip2@=1.0.8%rocmcc@=6.3.1~debug~pic+shared build_system=generic arch=linux-sles15-zen3
module load bzip2/1.0.8-rocmcc-6.3.1-u4w6woy
# pkgconf@=2.2.0%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load pkgconf/2.2.0-rocmcc-6.3.1-zuhzde2
# ncurses@=6.5%rocmcc@=6.3.1~symlinks+termlib abi=none build_system=autotools patches=7a351bc arch=linux-sles15-zen3
module load ncurses/6.5-rocmcc-6.3.1-4dt7xfj
# readline@=8.2%rocmcc@=6.3.1 build_system=autotools patches=bbf97f1 arch=linux-sles15-zen3
module load readline/8.2-rocmcc-6.3.1-hsjqzow
# gdbm@=1.23%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load gdbm/1.23-rocmcc-6.3.1-s5d7ejr
# zlib-ng@=2.2.1%rocmcc@=6.3.1+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-sles15-zen3
module load zlib-ng/2.2.1-rocmcc-6.3.1-d2zifpp
# perl@=5.40.0%rocmcc@=6.3.1+cpanm+opcode+open+shared+threads build_system=generic arch=linux-sles15-zen3
module load perl/5.40.0-rocmcc-6.3.1-2zihsrz
# automake@=1.16.5%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load automake/1.16.5-rocmcc-6.3.1-ienwcxn
# findutils@=4.9.0%rocmcc@=6.3.1 build_system=autotools patches=440b954 arch=linux-sles15-zen3
module load findutils/4.9.0-rocmcc-6.3.1-doylhaz
# libsigsegv@=2.14%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load libsigsegv/2.14-rocmcc-6.3.1-ilsqgla
# m4@=1.4.19%rocmcc@=6.3.1+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-sles15-zen3
module load m4/1.4.19-rocmcc-6.3.1-mcpzhki
# libtool@=2.4.7%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load libtool/2.4.7-rocmcc-6.3.1-tvjldov
# gmp@=6.3.0%rocmcc@=6.3.1+cxx build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load gmp/6.3.0-rocmcc-6.3.1-qwd3slg
# metis@=5.1.0%rocmcc@=6.3.1~gdb~int64~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903 arch=linux-sles15-zen3
module load metis/5.1.0-rocmcc-6.3.1-iqwjyj4
# autoconf-archive@=2023.02.20%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load autoconf-archive/2023.02.20-rocmcc-6.3.1-t7qwzg2
# xz@=5.4.6%rocmcc@=6.3.1~pic build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load xz/5.4.6-rocmcc-6.3.1-mth7psn
# libxml2@=2.13.4%rocmcc@=6.3.1+pic~python+shared build_system=autotools arch=linux-sles15-zen3
module load libxml2/2.13.4-rocmcc-6.3.1-hwoe2qh
# pigz@=2.8%rocmcc@=6.3.1 build_system=makefile arch=linux-sles15-zen3
module load pigz/2.8-rocmcc-6.3.1-cy5cnw7
# zstd@=1.5.6%rocmcc@=6.3.1+programs build_system=makefile compression=none libs=shared,static arch=linux-sles15-zen3
module load zstd/1.5.6-rocmcc-6.3.1-cftojny
# tar@=1.34%rocmcc@=6.3.1 build_system=autotools zip=pigz arch=linux-sles15-zen3
module load tar/1.34-rocmcc-6.3.1-kfi7nsj
# gettext@=0.22.5%rocmcc@=6.3.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-sles15-zen3
module load gettext/0.22.5-rocmcc-6.3.1-3ck747y
# texinfo@=7.1%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load texinfo/7.1-rocmcc-6.3.1-327afxj
# mpfr@=4.2.1%rocmcc@=6.3.1 build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load mpfr/4.2.1-rocmcc-6.3.1-h5arply
# openblas@=0.3.28%rocmcc@=6.3.1~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile patches=d0b9276 symbol_suffix=none threads=none arch=linux-sles15-zen3
module load openblas/0.3.28-rocmcc-6.3.1-yx5h226
# suite-sparse@=7.7.0%rocmcc@=6.3.1~cuda~graphblas~openmp+pic build_system=generic arch=linux-sles15-zen3
module load suite-sparse/7.7.0-rocmcc-6.3.1-k5bpn4t
# resolve@=develop%rocmcc@=6.3.1+asan~cuda~ipo+klu+lusol+rocm+ubsan amdgpu_target=gfx90a build_system=cmake build_type=Release dev_path=/lustre/orion/scratch/nkouk/stf006/Codes/ReSolve generator=make arch=linux-sles15-zen3
## module load resolve/develop-rocmcc-6.3.1-uzusuxq
