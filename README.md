# ReSolve

ReSolve is a library of GPU-resident linear solver. It contains iterative and direct linear solvers designed to run on NVIDIA and AMD GPUs, as well as on CPU devices.

## Getting started

Dependencies:
- KLU, AMD and COLAMD libraries from SuiteSparse
- CUDA >= 11.4
- CMake >= 3.22

To build it:
```shell
$ git clone https://code.ornl.gov/peles/resolve.git
$ mkdir build && cd build
$ cmake ../resolve
$ make
```

## To install the library 
In the directory where you built the library run
```shell
$ make install
```

To change the install location please use the CMAkePresets.json file as mentioned in [test and deploy](#test-and-deploy)

To run it, download [test linear systems](https://github.com/NREL/opf_matrices/tree/master/acopf/activsg10k) and then edit script [`runResolve`](runResolve) to match locations of your linear systems and binary installation. The script will emulate nonlinear solver calling the linear solver repeatedly.

## To use the ReSolve library in your own project
Make sure Resolve library is installed (see above)

Below is an example CMakeList.txt file to use ReSolve library in your project
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

For all contributions to ReSolve please follow the [developer guidelines](CONTRIBUTING.md)



## Test and Deploy

ReSolve as a library is tested every merge request via Gitlab pipelines that execute various library tests including a test of ReSolve being consumed as package within an external project as mentioned in [Using ReSolve in your own Project](#to-use-the-resolve-library-in-your-own-project)

To test your own install of ReSolve simply cd into your ReSolve build directory and run 

```shell
$ make test
```

or

```shell
$ ctest
```

Below is an example of what a functional ReSolve build will ouput on Passing tests

```text
Test project /people/dane678/resolve/build
    Start 1: resolve_consume
1/9 Test #1: resolve_consume ..................   Passed   16.51 sec
    Start 2: klu_klu_test
2/9 Test #2: klu_klu_test .....................   Passed    1.04 sec
    Start 3: klu_rf_test
3/9 Test #3: klu_rf_test ......................   Passed    1.04 sec
    Start 4: klu_rf_fgmres_test
4/9 Test #4: klu_rf_fgmres_test ...............   Passed    3.14 sec
    Start 5: klu_glu_test
5/9 Test #5: klu_glu_test .....................   Passed    1.06 sec
    Start 6: matrix_test
6/9 Test #6: matrix_test ......................   Passed    0.03 sec
    Start 7: matrix_handler_test
7/9 Test #7: matrix_handler_test ..............   Passed    0.97 sec
    Start 8: vector_handler_test
8/9 Test #8: vector_handler_test ..............   Passed    0.98 sec
    Start 9: logger_test
9/9 Test #9: logger_test ......................   Passed    0.03 sec
```

### Important Notes

You can find default Cmake Configurations in the CMakePresets.json file, which allows for easy switching between different CMake Configs. To create your own CMake Configuration we encourage you to utlize a CmakeUserPresets.json file. To learn more about cmake-presets please checkout the cmake [docs](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) 

For example if you wanted to build and install ReSolve on a High Performance Computing Cluster such as PNNL's Deception or ORNL's Ascent we encourage you to utilize our cluster preset. Using this preset will set CMAKE_INSTALL_PREFIX to an install folder. To use this preset simply call the preset flag in the cmake build step. 

```shell
cmake -B build --preset cluster
```

## Support
For any questions or to report a bug please submit a [GitHub issue](https://github.com/ORNL/ReSolve/issues). 

## Authors and acknowledgment
Primary authors of this project are Kasia &#346;wirydowicz <kasia.swirydowicz@pnnl.gov> and Slaven Peles <peless@ornl.gov>.

Multiple people contributed to the code developement (listed in alphabetic ortder):
- Ryan Danehy
- Jaelyn Litzinger
- Cameron Rutherford

Development of this coede was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations—the Office of Science and the National Nuclear Security Administration—responsible for the planning and preparation of a capable exascale ecosystem—including software, applications, hardware, advanced system engineering, and early testbed platforms—to support the nation's exascale computing imperative.

## License
Copyright &copy; 2023, UT-Battelle, LLC, and Battelle Memorial Institute.

ReSolve is a free software distributed under a BSD-style license. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details. All new contributions to ReSolve must be made under the smae licensing terms.
