module use -a /qfs/projects/exasgd/resolve/spack-ci/install/modules/linux-centos7-zen2
# cmake@=3.26.3%gcc@=9.1.0~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-centos7-zen2
module load cmake/3.26.3-gcc-9.1.0-idkqqqy
# cuda@=11.4%gcc@=9.1.0~allow-unsupported-compilers~dev build_system=generic arch=linux-centos7-zen2
module load cuda/11.4-gcc-9.1.0-7yrawjd
# gmake@=4.4.1%gcc@=9.1.0~guile build_system=autotools arch=linux-centos7-zen2
module load gmake/4.4.1-gcc-9.1.0-dbbnctg
# libiconv@=1.17%gcc@=9.1.0 build_system=autotools libs=shared,static arch=linux-centos7-zen2
module load libiconv/1.17-gcc-9.1.0-uewhnjd
# diffutils@=3.9%gcc@=9.1.0 build_system=autotools arch=linux-centos7-zen2
module load diffutils/3.9-gcc-9.1.0-3lfno34
# libsigsegv@=2.14%gcc@=9.1.0 build_system=autotools arch=linux-centos7-zen2
module load libsigsegv/2.14-gcc-9.1.0-5xah2f5
# m4@=1.4.19%gcc@=9.1.0+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-centos7-zen2
module load m4/1.4.19-gcc-9.1.0-k2ejaek
# perl@=5.26.0%gcc@=9.1.0+cpanm+opcode+open+shared+threads build_system=generic patches=0eac10e,8cf4302 arch=linux-centos7-zen2
module load perl/5.26.0-gcc-9.1.0-cw32fex
# autoconf@=2.69%gcc@=9.1.0 build_system=autotools patches=35c4492,7793209,a49dd5b arch=linux-centos7-zen2
module load autoconf/2.69-gcc-9.1.0-wpohlxq
# automake@=1.16.5%gcc@=9.1.0 build_system=autotools arch=linux-centos7-zen2
module load automake/1.16.5-gcc-9.1.0-nkzx2bw
# libtool@=2.4.7%gcc@=9.1.0 build_system=autotools arch=linux-centos7-zen2
module load libtool/2.4.7-gcc-9.1.0-dt4ss3b
# gmp@=6.2.1%gcc@=9.1.0+cxx build_system=autotools libs=shared,static patches=69ad2e2 arch=linux-centos7-zen2
module load gmp/6.2.1-gcc-9.1.0-dusqe6a
# metis@=5.1.0%gcc@=9.1.0~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-centos7-zen2
module load metis/5.1.0-gcc-9.1.0-5d5fnbf
# autoconf-archive@=2023.02.20%gcc@=9.1.0 build_system=autotools arch=linux-centos7-zen2
module load autoconf-archive/2023.02.20-gcc-9.1.0-q6tniqh
# bzip2@=1.0.8%gcc@=9.1.0~debug~pic+shared build_system=generic arch=linux-centos7-zen2
module load bzip2/1.0.8-gcc-9.1.0-fuwnbfz
# pkgconf@=1.9.5%gcc@=9.1.0 build_system=autotools arch=linux-centos7-zen2
module load pkgconf/1.9.5-gcc-9.1.0-b57m2xn
# xz@=5.4.1%gcc@=9.1.0~pic build_system=autotools libs=shared,static arch=linux-centos7-zen2
module load xz/5.4.1-gcc-9.1.0-c2ivx57
# zlib-ng@=2.1.3%gcc@=9.1.0+compat+opt build_system=autotools patches=299b958,ae9077a,b692621 arch=linux-centos7-zen2
module load zlib-ng/2.1.3-gcc-9.1.0-dzn64a6
# libxml2@=2.10.3%gcc@=9.1.0+pic~python+shared build_system=autotools arch=linux-centos7-zen2
module load libxml2/2.10.3-gcc-9.1.0-vfdylno
# ncurses@=6.4%gcc@=9.1.0~symlinks+termlib abi=none build_system=autotools arch=linux-centos7-zen2
module load ncurses/6.4-gcc-9.1.0-l2xyrw7
# pigz@=2.7%gcc@=9.1.0 build_system=makefile arch=linux-centos7-zen2
module load pigz/2.7-gcc-9.1.0-jvo2xwt
# zstd@=1.5.5%gcc@=9.1.0+programs build_system=makefile compression=none libs=shared,static arch=linux-centos7-zen2
module load zstd/1.5.5-gcc-9.1.0-jwdugup
# tar@=1.34%gcc@=9.1.0 build_system=autotools zip=pigz arch=linux-centos7-zen2
module load tar/1.34-gcc-9.1.0-cq6qrd3
# gettext@=0.21.1%gcc@=9.1.0+bzip2+curses+git~libunistring+libxml2+tar+xz build_system=autotools arch=linux-centos7-zen2
module load gettext/0.21.1-gcc-9.1.0-hynuimo
# texinfo@=7.0.3%gcc@=9.1.0 build_system=autotools arch=linux-centos7-zen2
module load texinfo/7.0.3-gcc-9.1.0-oygvjgv
# mpfr@=4.2.0%gcc@=9.1.0 build_system=autotools libs=shared,static arch=linux-centos7-zen2
module load mpfr/4.2.0-gcc-9.1.0-3jeasnm
# openblas@=0.3.23%gcc@=9.1.0~bignuma~consistent_fpcsr+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-centos7-zen2
module load openblas/0.3.23-gcc-9.1.0-4zbtr2v
# suite-sparse@=5.13.0%gcc@=9.1.0+cuda~graphblas~openmp+pic build_system=generic arch=linux-centos7-zen2
module load suite-sparse/5.13.0-gcc-9.1.0-pszalbf
# resolve@=develop%gcc@=9.1.0+cuda~ipo+klu build_system=cmake build_type=Release cuda_arch=60,70,75,80 dev_path=/people/ruth521/projects/resolve generator=make arch=linux-centos7-zen2
## module load resolve/develop-gcc-9.1.0-x6mhz5t
