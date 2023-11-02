module use -a /qfs/projects/exasgd/resolve/spack-ci/install/modules/linux-centos7-zen
# curl@=7.29.0%gcc@=8.4.0~gssapi~ldap~libidn2~librtmp~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl arch=linux-centos7-zen
module load curl/7.29.0-gcc-8.4.0-3emq5yx
# gmake@=4.4.1%gcc@=8.4.0~guile build_system=generic arch=linux-centos7-zen
module load gmake/4.4.1-gcc-8.4.0-l7nyr34
# pkgconf@=1.9.5%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load pkgconf/1.9.5-gcc-8.4.0-733ltud
# ncurses@=6.4%gcc@=8.4.0~symlinks+termlib abi=none build_system=autotools arch=linux-centos7-zen
module load ncurses/6.4-gcc-8.4.0-gwo76of
# zlib-ng@=2.1.4%gcc@=8.4.0+compat+opt build_system=autotools arch=linux-centos7-zen
module load zlib-ng/2.1.4-gcc-8.4.0-feah6zt
# cmake@=3.27.7%gcc@=8.4.0~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-centos7-zen
module load cmake/3.27.7-gcc-8.4.0-rmou7zf
# gmake@=4.4.1%clang@=16.0.0-rocm5.6.0 cxxflags="--gcc-toolchain=/share/apps/gcc/8.4.0" ~guile build_system=generic arch=linux-centos7-zen
module load gmake/4.4.1-clang-16.0.0-rocm5.6.0-6c7b35p
# python@=3.9.12%gcc@=8.4.0+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~nis~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic patches=0d98e93,4c24573,ebdca64,f2fd060 arch=linux-centos7-zen
module load python/3.9.12-gcc-8.4.0-ob2n5zs
# re2c@=2.2%gcc@=8.4.0 build_system=generic arch=linux-centos7-zen
module load re2c/2.2-gcc-8.4.0-zmj4cst
# ninja@=1.11.1%gcc@=8.4.0+re2c build_system=generic arch=linux-centos7-zen
module load ninja/1.11.1-gcc-8.4.0-ofxvwff
# z3@=4.11.2%gcc@=8.4.0~gmp~ipo~python build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load z3/4.11.2-gcc-8.4.0-363odap
# llvm-amdgpu@=5.6.1%gcc@=8.4.0~ipo~link_llvm_dylib~llvm_dylib~openmp+rocm-device-libs build_system=cmake build_type=Release generator=ninja patches=a08bbe1,b66529f,d35aec9 arch=linux-centos7-zen
module load llvm-amdgpu/5.6.1-gcc-8.4.0-vy3wrnq
# rocm-core@=5.6.1%gcc@=8.4.0~ipo build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load rocm-core/5.6.1-gcc-8.4.0-llv2yv4
# rocm-cmake@=5.6.1%gcc@=8.4.0~ipo build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load rocm-cmake/5.6.1-gcc-8.4.0-klwq5kk
# comgr@=5.6.1%gcc@=8.4.0~ipo build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load comgr/5.6.1-gcc-8.4.0-yl7z2re
# mesa@=23.0.2%gcc@=8.4.0+glx+llvm+opengl~opengles+osmesa~strip build_system=meson buildtype=release default_library=shared arch=linux-centos7-zen
module load mesa/23.0.2-gcc-8.4.0-xffioaq
# glx@=1.4%gcc@=8.4.0 build_system=bundle arch=linux-centos7-zen
module load glx/1.4-gcc-8.4.0-vh5g6sx
# hipify-clang@=5.6.1%gcc@=8.4.0~ipo build_system=cmake build_type=Release generator=make patches=54b8b39 arch=linux-centos7-zen
module load hipify-clang/5.6.1-gcc-8.4.0-e3jea5v
# libiconv@=1.17%gcc@=8.4.0 build_system=autotools libs=shared,static arch=linux-centos7-zen
module load libiconv/1.17-gcc-8.4.0-o2hwfiz
# diffutils@=3.9%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load diffutils/3.9-gcc-8.4.0-7ceszkk
# bzip2@=1.0.8%gcc@=8.4.0~debug~pic+shared build_system=generic arch=linux-centos7-zen
module load bzip2/1.0.8-gcc-8.4.0-on73m5o
# xz@=5.4.1%gcc@=8.4.0~pic build_system=autotools libs=shared,static arch=linux-centos7-zen
module load xz/5.4.1-gcc-8.4.0-v5kymdq
# libxml2@=2.10.3%gcc@=8.4.0+pic~python+shared build_system=autotools arch=linux-centos7-zen
module load libxml2/2.10.3-gcc-8.4.0-6mgqxiy
# pigz@=2.7%gcc@=8.4.0 build_system=makefile arch=linux-centos7-zen
module load pigz/2.7-gcc-8.4.0-btbzuey
# zstd@=1.5.5%gcc@=8.4.0+programs build_system=makefile compression=none libs=shared,static arch=linux-centos7-zen
module load zstd/1.5.5-gcc-8.4.0-3ets7dy
# tar@=1.34%gcc@=8.4.0 build_system=autotools zip=pigz arch=linux-centos7-zen
module load tar/1.34-gcc-8.4.0-atzwdgy
# gettext@=0.22.3%gcc@=8.4.0+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-centos7-zen
module load gettext/0.22.3-gcc-8.4.0-m33ujza
# libsigsegv@=2.14%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libsigsegv/2.14-gcc-8.4.0-gzna4n3
# m4@=1.4.19%gcc@=8.4.0+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-centos7-zen
module load m4/1.4.19-gcc-8.4.0-bwzchwl
# elfutils@=0.189%gcc@=8.4.0~debuginfod+exeprefix+nls build_system=autotools arch=linux-centos7-zen
module load elfutils/0.189-gcc-8.4.0-23kjwto
# libtool@=2.4.7%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libtool/2.4.7-gcc-8.4.0-2bmpsy4
# util-macros@=1.19.3%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load util-macros/1.19.3-gcc-8.4.0-64inrmm
# libpciaccess@=0.17%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libpciaccess/0.17-gcc-8.4.0-sh2c4la
# libpthread-stubs@=0.4%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libpthread-stubs/0.4-gcc-8.4.0-kcav646
# py-pip@=23.1.2%gcc@=8.4.0 build_system=generic arch=linux-centos7-zen
module load py-pip/23.1.2-gcc-8.4.0-yajovh7
# py-wheel@=0.41.2%gcc@=8.4.0 build_system=generic arch=linux-centos7-zen
module load py-wheel/0.41.2-gcc-8.4.0-dkkw2va
# py-setuptools@=68.0.0%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-setuptools/68.0.0-gcc-8.4.0-ihu4sfq
# meson@=1.2.2%gcc@=8.4.0 build_system=python_pip patches=0f0b1bd,ae59765 arch=linux-centos7-zen
module load meson/1.2.2-gcc-8.4.0-vcdwjmb
# libdrm@=2.4.115%gcc@=8.4.0~docs build_system=generic arch=linux-centos7-zen
module load libdrm/2.4.115-gcc-8.4.0-6h77lxh
# perl@=5.26.0%gcc@=8.4.0+cpanm+opcode+open+shared+threads build_system=generic patches=0eac10e,8cf4302 arch=linux-centos7-zen
module load perl/5.26.0-gcc-8.4.0-6tdzqfd
# autoconf@=2.69%gcc@=8.4.0 build_system=autotools patches=35c4492,7793209,a49dd5b arch=linux-centos7-zen
module load autoconf/2.69-gcc-8.4.0-dcrbb7h
# automake@=1.16.5%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load automake/1.16.5-gcc-8.4.0-tvi3cks
# numactl@=2.0.14%gcc@=8.4.0 build_system=autotools patches=4e1d78c,62fc8a8,ff37630 arch=linux-centos7-zen
module load numactl/2.0.14-gcc-8.4.0-7mpcwqq
# hsakmt-roct@=5.6.1%gcc@=8.4.0~ipo+shared build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load hsakmt-roct/5.6.1-gcc-8.4.0-4on3xib
# hsa-rocr-dev@=5.6.1%gcc@=8.4.0~image~ipo+shared build_system=cmake build_type=Release generator=make patches=9267179 arch=linux-centos7-zen
module load hsa-rocr-dev/5.6.1-gcc-8.4.0-tdlpv7w
# perl-file-which@=1.27%gcc@=8.4.0 build_system=perl arch=linux-centos7-zen
module load perl-file-which/1.27-gcc-8.4.0-nix64yx
# perl-module-build@=0.4232%gcc@=8.4.0 build_system=perl arch=linux-centos7-zen
module load perl-module-build/0.4232-gcc-8.4.0-ayed35p
# perl-uri-encode@=1.1.1%gcc@=8.4.0 build_system=perl arch=linux-centos7-zen
module load perl-uri-encode/1.1.1-gcc-8.4.0-biqataj
# py-ply@=3.11%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-ply/3.11-gcc-8.4.0-creftnl
# py-cppheaderparser@=2.7.4%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-cppheaderparser/2.7.4-gcc-8.4.0-nw7554i
# rocminfo@=5.6.1%gcc@=8.4.0~ipo build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load rocminfo/5.6.1-gcc-8.4.0-5shaxxj
# roctracer-dev-api@=5.6.1%gcc@=8.4.0 build_system=generic arch=linux-centos7-zen
module load roctracer-dev-api/5.6.1-gcc-8.4.0-gbaoh25
# hip@=5.6.1%gcc@=8.4.0~cuda~ipo+rocm build_system=cmake build_type=Release generator=make patches=aee7249,c2ee21c,e73e91b arch=linux-centos7-zen
module load hip/5.6.1-gcc-8.4.0-zpa2j7f
# msgpack-c@=3.1.1%gcc@=8.4.0~ipo build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load msgpack-c/3.1.1-gcc-8.4.0-buxbznu
# procps@=4.0.4%gcc@=8.4.0+nls build_system=autotools arch=linux-centos7-zen
module load procps/4.0.4-gcc-8.4.0-gyn6his
# py-joblib@=1.2.0%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-joblib/1.2.0-gcc-8.4.0-ukcd432
# py-cython@=0.29.36%gcc@=8.4.0 build_system=python_pip patches=c4369ad arch=linux-centos7-zen
module load py-cython/0.29.36-gcc-8.4.0-5f4zyzb
# py-msgpack@=1.0.5%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-msgpack/1.0.5-gcc-8.4.0-2xh5udm
# libyaml@=0.2.5%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libyaml/0.2.5-gcc-8.4.0-hidc7bw
# py-pyyaml@=6.0%gcc@=8.4.0+libyaml build_system=python_pip arch=linux-centos7-zen
module load py-pyyaml/6.0-gcc-8.4.0-4mdsdw2
# py-distlib@=0.3.7%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-distlib/0.3.7-gcc-8.4.0-f25ay4b
# py-editables@=0.3%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-editables/0.3-gcc-8.4.0-hrmamrk
# py-flit-core@=3.9.0%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-flit-core/3.9.0-gcc-8.4.0-q3yng6k
# py-packaging@=23.1%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-packaging/23.1-gcc-8.4.0-7krugqt
# py-pathspec@=0.11.1%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-pathspec/0.11.1-gcc-8.4.0-vm5freh
# git@=2.42.0%gcc@=8.4.0+man+nls+perl+subtree~svn~tcltk build_system=autotools arch=linux-centos7-zen
module load git/2.42.0-gcc-8.4.0-k5crf2q
# py-tomli@=2.0.1%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-tomli/2.0.1-gcc-8.4.0-m4gh2nb
# py-typing-extensions@=4.8.0%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-typing-extensions/4.8.0-gcc-8.4.0-ovqdpbs
# py-setuptools-scm@=7.1.0%gcc@=8.4.0+toml build_system=python_pip arch=linux-centos7-zen
module load py-setuptools-scm/7.1.0-gcc-8.4.0-hqzn5lb
# py-pluggy@=1.0.0%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-pluggy/1.0.0-gcc-8.4.0-lqpf66l
# py-calver@=2022.6.26%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-calver/2022.6.26-gcc-8.4.0-pm6rj2c
# py-trove-classifiers@=2023.8.7%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-trove-classifiers/2023.8.7-gcc-8.4.0-iy66qnh
# py-hatchling@=1.18.0%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-hatchling/1.18.0-gcc-8.4.0-bjpjiiq
# py-hatch-vcs@=0.3.0%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-hatch-vcs/0.3.0-gcc-8.4.0-hc6rq3a
# py-filelock@=3.12.4%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-filelock/3.12.4-gcc-8.4.0-rzqmlrq
# py-platformdirs@=3.10.0%gcc@=8.4.0~wheel build_system=python_pip arch=linux-centos7-zen
module load py-platformdirs/3.10.0-gcc-8.4.0-6hnyp7h
# py-virtualenv@=20.24.5%gcc@=8.4.0 build_system=python_pip arch=linux-centos7-zen
module load py-virtualenv/20.24.5-gcc-8.4.0-h4mzkzl
# rocblas@=5.6.1%gcc@=8.4.0~ipo+tensile amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load rocblas/5.6.1-gcc-8.4.0-arsno2b
# fmt@=10.1.1%gcc@=8.4.0~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make arch=linux-centos7-zen
module load fmt/10.1.1-gcc-8.4.0-4d5ehr5
# rocprim@=5.6.1%gcc@=8.4.0~ipo amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load rocprim/5.6.1-gcc-8.4.0-nu465tt
# rocsparse@=5.6.1%gcc@=8.4.0~ipo~test amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load rocsparse/5.6.1-gcc-8.4.0-wtmfgyn
# rocsolver@=5.6.1%gcc@=8.4.0~ipo+optimal amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load rocsolver/5.6.1-gcc-8.4.0-wlgpkqj
# roctracer-dev@=5.6.1%gcc@=8.4.0~ipo~rocm build_system=cmake build_type=Release generator=make arch=linux-centos7-zen
module load roctracer-dev/5.6.1-gcc-8.4.0-lilld4h
# libiconv@=1.17%gcc@=8.4.0 build_system=autotools libs=shared,static arch=linux-centos7-zen
module load libiconv/1.17-gcc-8.4.0-wfdnlg6
# diffutils@=3.9%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load diffutils/3.9-gcc-8.4.0-qh566r6
# libsigsegv@=2.14%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libsigsegv/2.14-gcc-8.4.0-iutj4de
# m4@=1.4.19%gcc@=8.4.0+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-centos7-zen
module load m4/1.4.19-gcc-8.4.0-x7ktvaf
# perl@=5.26.0%gcc@=8.4.0+cpanm+opcode+open+shared+threads build_system=generic patches=0eac10e,8cf4302 arch=linux-centos7-zen
module load perl/5.26.0-gcc-8.4.0-h324qox
# autoconf@=2.69%gcc@=8.4.0 build_system=autotools patches=35c4492,7793209,a49dd5b arch=linux-centos7-zen
module load autoconf/2.69-gcc-8.4.0-npluk5j
# automake@=1.16.5%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load automake/1.16.5-gcc-8.4.0-tgloywk
# libtool@=2.4.7%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load libtool/2.4.7-gcc-8.4.0-gs6gyy3
# gmp@=6.2.1%gcc@=8.4.0+cxx build_system=autotools libs=shared,static patches=69ad2e2 arch=linux-centos7-zen
module load gmp/6.2.1-gcc-8.4.0-ythx4o2
# pkgconf@=1.9.5%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load pkgconf/1.9.5-gcc-8.4.0-kl4sdjo
# nghttp2@=1.52.0%gcc@=8.4.0 build_system=autotools arch=linux-centos7-zen
module load nghttp2/1.52.0-gcc-8.4.0-pqmjl5g
# ca-certificates-mozilla@=2023-05-30%gcc@=8.4.0 build_system=generic arch=linux-centos7-zen
module load ca-certificates-mozilla/2023-05-30-gcc-8.4.0-txgcsig
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
# resolve@=develop%clang@=16.0.0-rocm5.6.0 cxxflags="--gcc-toolchain=/share/apps/gcc/8.4.0" ~cuda~ipo+klu+rocm amdgpu_target=gfx908 build_system=cmake build_type=Release dev_path=/people/ruth521/projects/resolve generator=make arch=linux-centos7-zen
## module load resolve/develop-clang-16.0.0-rocm5.6.0-6kaaut4
