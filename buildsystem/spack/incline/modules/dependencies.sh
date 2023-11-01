module use -a /qfs/projects/exasgd/resolve/spack-ci/install/modules/linux-centos7-zen
# pkgconf@=1.9.5%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load pkgconf/1.9.5-gcc-8.4.0-kl4sdjo
# nghttp2@=1.52.0%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load nghttp2/1.52.0-gcc-8.4.0-pqmjl5g
# ca-certificates-mozilla@=2023-05-30%gcc@=8.4.0 build_system=generic arch=linux-centos7-zen
module load ca-certificates-mozilla/2023-05-30-gcc-8.4.0-txgcsig
# perl@=5.26.0%gcc@=8.4.0+cpanm+opcode+open+shared+threads build_system=generic patches=0eac10e,8cf4302 arch=linux-centos7-zen
module load perl/5.26.0-gcc-8.4.0-h324qox
# zlib-ng@=2.1.3%gcc@=8.4.0+compat+opt build_system=autotools patches=299b958,ae9077a,b692621 arch=linux-centos7-zen
module load zlib-ng/2.1.3-gcc-8.4.0-44tydhr
# openssl@=3.1.3%gcc@=8.4.0~docs+shared build_system=generic certs=mozilla arch=linux-centos7-zen
module load openssl/3.1.3-gcc-8.4.0-46yttzm
# curl@=8.4.0%gcc@=8.4.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl arch=linux-centos7-zen
module load curl/8.4.0-gcc-8.4.0-g2rrs23
# ncurses@=6.4%gcc@=8.4.0~symlinks+termlib abi=none build_system=autotools arch=linux-centos7-zen
module load ncurses/6.4-gcc-8.4.0-jt7rpqq
# cmake@=3.27.7%gcc@=8.4.0~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-centos7-zen
module load cmake/3.27.7-gcc-8.4.0-tu2rruq
# gmake@=4.4.1%gcc@=8.4.0~guile build_system=generic arch=linux-centos7-zen
module load gmake/4.4.1-gcc-8.4.0-l7nyr34
# libiconv@=1.17%gcc@=8.4.0 build_system=autotools libs=shared,static arch=linux-centos7-zen
module load libiconv/1.17-gcc-8.4.0-wfdnlg6
# diffutils@=3.9%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load diffutils/3.9-gcc-8.4.0-qh566r6
# libsigsegv@=2.14%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libsigsegv/2.14-gcc-8.4.0-iutj4de
# m4@=1.4.19%gcc@=8.4.0+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-centos7-zen
module load m4/1.4.19-gcc-8.4.0-x7ktvaf
# autoconf@=2.69%gcc@=8.4.0 build_system=autotools patches=35c4492,7793209,a49dd5b arch=linux-centos7-zen
module load autoconf/2.69-gcc-8.4.0-npluk5j
# automake@=1.16.5%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load automake/1.16.5-gcc-8.4.0-tgloywk
# libtool@=2.4.7%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libtool/2.4.7-gcc-8.4.0-gs6gyy3
# gmp@=6.2.1%gcc@=8.4.0+cxx build_system=autotools libs=shared,static patches=69ad2e2 arch=linux-centos7-zen
module load gmp/6.2.1-gcc-8.4.0-ythx4o2
# gmake@=4.4.1%gcc@=8.4.0~guile build_system=autotools arch=linux-centos7-zen
module load gmake/4.4.1-gcc-8.4.0-f23wik2
# metis@=5.1.0%gcc@=8.4.0~gdb~int64~ipo~real64+shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-centos7-zen
module load metis/5.1.0-gcc-8.4.0-gsllf6a
# autoconf-archive@=2023.02.20%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load autoconf-archive/2023.02.20-gcc-8.4.0-ox4hxoe
# bzip2@=1.0.8%gcc@=8.4.0~debug~pic+shared build_system=generic arch=linux-centos7-zen
module load bzip2/1.0.8-gcc-8.4.0-3uzyl47
# xz@=5.4.1%gcc@=8.4.0~pic build_system=autotools libs=shared,static arch=linux-centos7-zen
module load xz/5.4.1-gcc-8.4.0-dwmuagy
# libxml2@=2.10.3%gcc@=8.4.0+pic~python+shared build_system=autotools arch=linux-centos7-zen
module load libxml2/2.10.3-gcc-8.4.0-2hu4ayt
# pigz@=2.7%gcc@=8.4.0 build_system=makefile arch=linux-centos7-zen
module load pigz/2.7-gcc-8.4.0-lu7bjb6
# zstd@=1.5.5%gcc@=8.4.0+programs build_system=makefile compression=none libs=shared,static arch=linux-centos7-zen
module load zstd/1.5.5-gcc-8.4.0-z7jmyvw
# tar@=1.34%gcc@=8.4.0 build_system=autotools zip=pigz arch=linux-centos7-zen
module load tar/1.34-gcc-8.4.0-wcgempy
# gettext@=0.22.3%gcc@=8.4.0+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-centos7-zen
module load gettext/0.22.3-gcc-8.4.0-f7dl6un
# texinfo@=7.0.3%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load texinfo/7.0.3-gcc-8.4.0-jma4obj
# mpfr@=4.2.0%gcc@=8.4.0 build_system=autotools libs=shared,static arch=linux-centos7-zen
module load mpfr/4.2.0-gcc-8.4.0-cjhi2el
# openblas@=0.3.24%gcc@=8.4.0~bignuma~consistent_fpcsr+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-centos7-zen
module load openblas/0.3.24-gcc-8.4.0-4ei4hpg
# suite-sparse@=5.13.0%gcc@=8.4.0~cuda~graphblas~openmp+pic build_system=generic arch=linux-centos7-zen
module load suite-sparse/5.13.0-gcc-8.4.0-ivey23b
# resolve@=develop%gcc@=8.4.0~cuda~ipo+klu build_system=cmake build_type=Release dev_path=/people/svcexasgd/gitlab/24143/spack_incline generator=make arch=linux-centos7-zen
## module load resolve/develop-gcc-8.4.0-l7tspub
