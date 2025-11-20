<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/ReSolve-Logo-Dark.png" width="350" valign="middle">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/ReSolve-Logo-Light.png" width="350" valign="middle">
  <img alt="Re::Solve">
</picture>

<h2>

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/ORNL/ReSolve/blob/develop/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/resolve/badge/?version=develop)](https://resolve.readthedocs.io/en/develop/?badge=develop)
[![Spack CPU Build](https://github.com/ORNL/ReSolve/actions/workflows/spack_cpu_build.yaml/badge.svg?event=pull_request)](https://github.com/ORNL/ReSolve/actions)

</h2>

Re::Solve is a library of GPU-resident linear solvers. It contains iterative
and direct solvers designed to run on NVIDIA and AMD GPUs, as well as on CPU
devices.

The User Guide and developer's documentation is available
[online](https://resolve.readthedocs.io/en/latest/), including Doxygen-generated
[source code documentation](https://resolve.readthedocs.io/en/latest/doxygen/html/index.html).


## Getting started

Dependencies:
- CMake >= 3.22
- KLU, AMD and COLAMD libraries from SuiteSparse >= 5.0 (optional)
- CUDA >= 11.4 (optional)
- HIP/ROCm >= 6.4 (optional)

To build it:
```shell
$ git clone https://github.com/ORNL/ReSolve.git
$ mkdir build && cd build
$ cmake ../ReSolve
$ make
```

## To install the library 
In the directory where you built the library run
```shell
$ make install
```

To change the install location please use the CMAkePresets.json file as
mentioned in [test and deploy](#test-and-deploy)

To run it, download [test linear systems](https://github.com/NREL/opf_matrices/tree/master/acopf/activsg10k)
and then try some of Re::Solve's [examples](https://github.com/ORNL/ReSolve/tree/develop/examples).
The examples will emulate nonlinear solver calling the linear solver repeatedly.

### Create your own CMake configurations

You can find default CMake Configurations in the CMakePresets.json file, which
allows for easy switching between different CMake configurations. To create
your own CMake configuration we encourage you to utlize a CmakeUserPresets.json
file. To learn more about cmake-presets please checkout the cmake
[docs](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) 

For example if you wanted to build and install Re::Solve on a High Performance
Computing Cluster such as PNNL's Deception or ORNL's Ascent we encourage you to
utilize our cluster preset. Using this preset will set CMAKE_INSTALL_PREFIX to
an install folder. To use this preset simply call the preset flag in the cmake
build step. 

```shell
cmake -B build --preset cluster
```

## Test and Deploy

Re::Solve as a library is tested every merge request via Gitlab pipelines that
execute various library tests including a test of Re::Solve being consumed as
package within an external project as mentioned in
[Using Re::Solve in your own Project](#to-use-the-resolve-library-in-your-own-project)

To test your own install of Re::Solve simply run from your Re::Solve build
directory 
```shell
$ make test
```
After you `make install` you can test your installation by running
```shell
$ make test_install
```
from your build directory.


## To use the Re::Solve library in your own project
Make sure Resolve library is installed (see above)

Below is an example CMakeList.txt file to use Re::Solve library in your project
```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(ReSolve CONFIG 
  PATHS ${ReSolve_DIR} ${ReSolve_DIR}/share/resolve/cmake
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  REQUIRED)

# Build your executable 
add_executable(my_app my_app.cpp)
target_link_libraries(my_app PRIVATE ReSolve::ReSolve)
```


## Contributing

For all contributions to Re::Solve please follow the
[developer guidelines](CONTRIBUTING.md)


## Support
For technical questions or to report a bug please submit a
[GitHub issue](https://github.com/ORNL/ReSolve/issues) or post a question on
[user mailing list](mailto:resolve-users@elist.ornl.gov).
For non-technical issues please contact
[Re::Solve developers](mailto:resolve-devel@elist.ornl.gov).

## Authors and acknowledgment
The primary authors of this project are Kasia &#346;wirydowicz, Slaven Peles, and Shaked Regev.

Re::Solve project would not be possible without significant contributions from
(in alphabetic order):
- Maksudul Alam
- Kaleb Brunhoeber
- Ryan Danehy
- Adham Ibrahim 
- Nicholson Koukpaizan
- Jaelyn Litzinger
- Phil Roth
- Cameron Rutherford

Development of this code was supported by the Exascale Computing Project (ECP),
Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations—the
Office of Science and the National Nuclear Security Administration—responsible
for the planning and preparation of a capable exascale ecosystem—including
software, applications, hardware, advanced system engineering, and early
testbed platforms—to support the nation's exascale computing imperative.

## License
Copyright &copy; 2023, UT-Battelle, LLC, and Battelle Memorial Institute.

Re::Solve is a free software distributed under a BSD-style license. See the
[LICENSE](LICENSE) and [NOTICE](NOTICE) files for details. All new
contributions to Re::Solve must be made under the same licensing terms.

**Please Note** If you are using Re::Solve with any third party libraries linked
in (e.g., KLU), be sure to review the respective license of the package as that
license may have more restrictive terms than the Re::Solve license.
