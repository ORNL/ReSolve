
CMake Build System
------------------

Our ``cmake`` folder contains some basic CMake modules that help manage resolve:

* ``cmake/FindKLU.cmake``: Our custom find module for KLU that we maintain
* ``cmake/ReSolveConfig.cmake.in``: Our custom config file that is used to generate the ``ReSolveConfig.cmake`` file that is installed with Re::Solve
* ``cmake/ReSolveFindCudaLibraries.cmake``: Our custom find module for CUDA libraries that we maintain to link in subset of cuda needed
* ``cmake/ReSolveFindHipLibraries.cmake``: Our custom find module for HIP/ROCm libraries that we maintain to link in subset of hip needed

Apart from that check out our main ``CMakeLists.txt`` file for our remaining build configuration.

We also export under the ``ReSolve::`` namespace in our installed CMake configuration for use with ``find_package`` as documented in our main ``README.md``.

Spack Package
-------------

Re::Solve can be built with 
`spack <https://github.com/spack/spack/tree/develop>`_, and contains support
for building Re::Solve with KLU and CUDA or HIP/ROCm support. 

We also have a custom ``spack`` folder/installation that contains our spack
submodule located in ``buildsystem/spack/spack``. This is used to build
Re::Solve on CI platforms, as well as support development of the spack package
as neccessary.

See the Quick How-To section below for more information on how to update the
spack package and typical workflows for building Re::Solve with spack on CI
platforms for testing.
