module use -a /lustre/orion/stf006/world-shared/nkouk/resolve/spack-install/modules/linux-sles15-zen3
# cmake@=3.27.9%rocmcc@=6.3.1~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-sles15-zen3
module load cmake/3.27.9-rocmcc-6.3.1-u2nxzzw
# glibc@=2.31%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load glibc/2.31-rocmcc-6.3.1-3lnhxla
# gmake@=4.4.1%rocmcc@=6.3.1~guile build_system=generic arch=linux-sles15-zen3
module load gmake/4.4.1-rocmcc-6.3.1-r3ngrru
# hip@=6.3.1%rocmcc@=6.3.1~cuda+rocm build_system=cmake build_type=Release generator=make patches=1f65dfe arch=linux-sles15-zen3
module load hip/6.3.1-rocmcc-6.3.1-ykh7piu
# hsa-rocr-dev@=6.3.1%rocmcc@=6.3.1~asan+image+shared build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load hsa-rocr-dev/6.3.1-rocmcc-6.3.1-xxxdnpr
# llvm-amdgpu@=6.3.1%rocmcc@=6.3.1~link_llvm_dylib~llvm_dylib+rocm-device-libs build_system=cmake build_type=Release generator=ninja patches=b4774ca arch=linux-sles15-zen3
module load llvm-amdgpu/6.3.1-rocmcc-6.3.1-vtiuud6
# rocblas@=6.3.1%rocmcc@=6.3.1+tensile amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocblas/6.3.1-rocmcc-6.3.1-3v25znh
# rocsolver@=6.3.1%rocmcc@=6.3.1+optimal amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocsolver/6.3.1-rocmcc-6.3.1-zt6tvxo
# rocsparse@=6.3.1%rocmcc@=6.3.1~test amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-sles15-zen3
module load rocsparse/6.3.1-rocmcc-6.3.1-f6dcxbw
# autoconf@=2.72%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load autoconf/2.72-rocmcc-6.3.1-iuqfc3b
# berkeley-db@=18.1.40%rocmcc@=6.3.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-sles15-zen3
module load berkeley-db/18.1.40-rocmcc-6.3.1-h6eg4c7
# libiconv@=1.17%rocmcc@=6.3.1 build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load libiconv/1.17-rocmcc-6.3.1-rd4vhmq
# diffutils@=3.10%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load diffutils/3.10-rocmcc-6.3.1-7l6sjdf
# bzip2@=1.0.8%rocmcc@=6.3.1~debug~pic+shared build_system=generic arch=linux-sles15-zen3
module load bzip2/1.0.8-rocmcc-6.3.1-wbzciku
# pkgconf@=2.2.0%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load pkgconf/2.2.0-rocmcc-6.3.1-kfyxruo
# ncurses@=6.5%rocmcc@=6.3.1~symlinks+termlib abi=none build_system=autotools patches=7a351bc arch=linux-sles15-zen3
module load ncurses/6.5-rocmcc-6.3.1-jrst3u7
# readline@=8.2%rocmcc@=6.3.1 build_system=autotools patches=bbf97f1 arch=linux-sles15-zen3
module load readline/8.2-rocmcc-6.3.1-ejxahic
# gdbm@=1.23%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load gdbm/1.23-rocmcc-6.3.1-fpdef6v
# zlib-ng@=2.1.6%rocmcc@=6.3.1+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-sles15-zen3
module load zlib-ng/2.1.6-rocmcc-6.3.1-ogcqdz6
# perl@=5.38.2%rocmcc@=6.3.1+cpanm+opcode+open+shared+threads build_system=generic patches=714e4d1 arch=linux-sles15-zen3
module load perl/5.38.2-rocmcc-6.3.1-2whd6aa
# automake@=1.16.5%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load automake/1.16.5-rocmcc-6.3.1-oiwl7pa
# findutils@=4.9.0%rocmcc@=6.3.1 build_system=autotools patches=440b954 arch=linux-sles15-zen3
module load findutils/4.9.0-rocmcc-6.3.1-hhqjtne
# libsigsegv@=2.14%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load libsigsegv/2.14-rocmcc-6.3.1-54llo7y
# m4@=1.4.19%rocmcc@=6.3.1+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-sles15-zen3
module load m4/1.4.19-rocmcc-6.3.1-6s73i4y
# libtool@=2.4.7%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load libtool/2.4.7-rocmcc-6.3.1-duohpys
# gmp@=6.3.0%rocmcc@=6.3.1+cxx build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load gmp/6.3.0-rocmcc-6.3.1-6mlfcrh
# metis@=5.1.0%rocmcc@=6.3.1~gdb~int64~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903 arch=linux-sles15-zen3
module load metis/5.1.0-rocmcc-6.3.1-7p6jagm
# autoconf-archive@=2023.02.20%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load autoconf-archive/2023.02.20-rocmcc-6.3.1-y5tjjor
# xz@=5.4.6%rocmcc@=6.3.1~pic build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load xz/5.4.6-rocmcc-6.3.1-ofz2owp
# libxml2@=2.10.3%rocmcc@=6.3.1+pic~python+shared build_system=autotools arch=linux-sles15-zen3
module load libxml2/2.10.3-rocmcc-6.3.1-ycqoioe
# pigz@=2.8%rocmcc@=6.3.1 build_system=makefile arch=linux-sles15-zen3
module load pigz/2.8-rocmcc-6.3.1-pwg4fa7
# zstd@=1.5.6%rocmcc@=6.3.1+programs build_system=makefile compression=none libs=shared,static arch=linux-sles15-zen3
module load zstd/1.5.6-rocmcc-6.3.1-4rcbzfe
# tar@=1.34%rocmcc@=6.3.1 build_system=autotools zip=pigz arch=linux-sles15-zen3
module load tar/1.34-rocmcc-6.3.1-k5s7r3a
# gettext@=0.22.5%rocmcc@=6.3.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-sles15-zen3
module load gettext/0.22.5-rocmcc-6.3.1-phtfuvo
# texinfo@=7.1%rocmcc@=6.3.1 build_system=autotools arch=linux-sles15-zen3
module load texinfo/7.1-rocmcc-6.3.1-fxp3zqc
# mpfr@=4.2.1%rocmcc@=6.3.1 build_system=autotools libs=shared,static arch=linux-sles15-zen3
module load mpfr/4.2.1-rocmcc-6.3.1-nny7r7l
# openblas@=0.3.26%rocmcc@=6.3.1~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-sles15-zen3
module load openblas/0.3.26-rocmcc-6.3.1-u7ecad6
# suite-sparse@=7.7.0%rocmcc@=6.3.1~cuda~graphblas~openmp+pic build_system=generic arch=linux-sles15-zen3
module load suite-sparse/7.7.0-rocmcc-6.3.1-tsd34uu
# resolve@=develop%rocmcc@=6.3.1~cuda~ipo+klu+lusol+rocm amdgpu_target=gfx90a build_system=cmake build_type=Release dev_path=/lustre/orion/scratch/nkouk/stf006/Codes/ReSolve generator=make arch=linux-sles15-zen3
## module load resolve/develop-rocmcc-6.3.1-y5fkoxv
