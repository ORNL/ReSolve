# ReSolve

ReSolve is a library of GPU-resident linear solver. It is very much work in progres. When completed, it will contain iterative and direct linear solvers designed to run on NVIDIA and AMD GPUs, as well as CPU devices.

## Getting started

Dependencies:
- KLU, AMD and COLAMD libraries from SuiteSparse
- CUDA >= 11.4
- CMake >= 3.20

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

Use the built-in continuous integration in GitLab.




## Support
For any questions or to report a bug please submit a [GitLab issue](https://code.ornl.gov/ecpcitest/exasgd/resolve/-/issues). 

## Authors and acknowledgment
Primary author of this project is Kasia &#346;wirydowicz <kasia.swirydowicz@pnnl.gov>.  

Development of this coede was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations—the Office of Science and the National Nuclear Security Administration—responsible for the planning and preparation of a capable exascale ecosystem—including software, applications, hardware, advanced system engineering, and early testbed platforms—to support the nation's exascale computing imperative.

## License
Copyright &copy; 2023, UT-Battelle, LLC, and Battelle Memorial Institute.

ReSolve is a free software distributed under a BSD-style license. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details. All new contributions to ReSolve must be made under the smae licensing terms.
