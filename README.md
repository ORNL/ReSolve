# ReSolve

ReSolve is a library of GPU-resident linear solver. It contains iterative and direct linear solvers designed to run on NVIDIA and AMD GPUs, as well as on CPU devices.

ReadTheDocs Documentation lives here https://resolve.readthedocs.io/en/develop/

## Getting started

Dependencies:
- KLU, AMD and COLAMD libraries from SuiteSparse >= 5.0
- CMake >= 3.22
- CUDA >= 11.4 (optional)
- HIP/ROCm >= 5.6 (optional)

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

### Pre-commit
Precommit is a framework for mmangaing and maintain multi-lanagueg pre-commit hooks. https://pre-commit.com/ 

Precommit is auto applied through CI to every pull request and every commit to the said pull request. Precommit checks for valid yaml files, end of file spaces, Syntax of C codeand etc. To see specifically what formating precommit looks for please check out our .clang-format and .cmake-format filed as well as our .pre-commit-config.yaml which show the specific precommit hooks being applied. 

### Running Pre-commit locally
To run pre-commit locally you first must install pre-commit and there are a couple ways to do this.

#### Prerequiste to installing pre-commit
It is recommended that you install pre-commit through python. Here we assume Python is aleadly installed if that is not the case please check out these [docs](https://wiki.python.org/moin/BeginnersGuide/Download) for how to install python. 

Next create a virtual environment with the following:
```shell
python -m venv /path/to/new/virtual/environment
```

Activate the new environment:
```shell
source /path/to/new/virtual/environment/bin/activate
```

 Now that we are within a virtual environment we can install pre-commit via;
```shell
pip install pre-commit
```

To install the hooks and have pre-commit run automatically before every commit run
```shell
pre-commit install --install-hooks
```

If you want to manually run all pre-commit hooks on a repository, run 
```shell
pre-commit run --all-files
```
To learn more about pre-commit usuage check out https://pre-commit.com/#usage.

## Test and Deploy

ReSolve as a library is tested every merge request via Gitlab pipelines that execute various library tests including a test of ReSolve being consumed as package within an external project as mentioned in [Using ReSolve in your own Project](#to-use-the-resolve-library-in-your-own-project)

To test your own install of ReSolve simply run from your ReSolve build directory 
```shell
$ make test
```
After you `make install` you can test your installation by running
```shell
$ make test_install
```
from your build directory.


### Important Notes

You can find default Cmake Configurations in the CMakePresets.json file, which allows for easy switching between different CMake Configs. To create your own CMake Configuration we encourage you to utlize a CmakeUserPresets.json file. To learn more about cmake-presets please checkout the cmake [docs](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) 

For example if you wanted to build and install ReSolve on a High Performance Computing Cluster such as PNNL's Deception or ORNL's Ascent we encourage you to utilize our cluster preset. Using this preset will set CMAKE_INSTALL_PREFIX to an install folder. To use this preset simply call the preset flag in the cmake build step. 

```shell
cmake -B build --preset cluster
```

## Support
For technical questions or to report a bug please submit a [GitHub issue](https://github.com/ORNL/ReSolve/issues).
For non-technical issues please contact Kasia &#346;wirydowicz <kasia.swirydowicz@pnnl.gov> or Slaven Peles <peless@ornl.gov>.

## Authors and acknowledgment
Primary authors of this project are Kasia &#346;wirydowicz and Slaven Peles.

ReSolve project would not be possible without significant contributions from (in alphabetic order):
- Maksudul Alam
- Ryan Danehy
- Nicholson Koukpaizan
- Jaelyn Litzinger
- Phil Roth
- Cameron Rutherford

Development of this code was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC,
a collaborative effort of two DOE organizations—the Office of Science and the National Nuclear Security
Administration—responsible for the planning and preparation of a capable exascale ecosystem—including software,
applications, hardware, advanced system engineering, and early testbed platforms—to support the nation's exascale
computing imperative.

## License
Copyright &copy; 2023, UT-Battelle, LLC, and Battelle Memorial Institute.

ReSolve is a free software distributed under a BSD-style license. See the
[LICENSE](LICENSE) and [NOTICE](NOTICE) files for details. All new
contributions to ReSolve must be made under the smae licensing terms.

**Please Note** If you are using ReSolve with any third party libraries linked
in (e.g., KLU), be sure to review the respective license of the package as that
license may have more restrictive terms than the ReSolve license.
