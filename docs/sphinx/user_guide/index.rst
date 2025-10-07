User Guide
====================

Getting started
---------------
ReSolve is maintained and developed at `Github <https://github.com/ORNL/ReSolve>`_.
It has following build dependencies:

* C++11 compliant compiler
* CMake >= 3.22
* KLU, AMD and COLAMD libraries from SuiteSparse >= 5.0 (optional)
* CUDA >= 11.4 (optional)
* HIP/ROCm >= 5.6 (optional)


To acquire and build Re::Solve it is as easy as:

.. code:: shell

   $ git clone https://code.ornl.gov/peles/resolve.git
   $ mkdir build && cd build
   $ cmake ../resolve
   $ make

Note if you don't include any of the optional dependencies, there will be
little functionality provided. You might want to use tool such as ``ccmake``
to enable features you need.

To install the library
----------------------

In the directory where you built the library run

.. code:: shell

   $ make install

To change the install location please use the CMakePresets.json file as mentioned in `test and deploy <#test-and-deploy>`__

To run it, download `test linear systems <https://github.com/NREL/opf_matrices/tree/master/acopf/activsg10k>`__
and then edit script |runResolve|_
to match locations of your linear systems and binary installation.
The script will emulate nonlinear solver calling the linear solver repeatedly.

To use the ReSolve library in your own project
----------------------------------------------

Make sure Resolve library is installed (see above)

Below is an example CMakeList.txt file to use ReSolve library in your project

.. code:: cmake

   cmake_minimum_required(VERSION 3.20)
   project(my_app LANGUAGES CXX)

   find_package(ReSolve CONFIG
     PATHS ${ReSolve_DIR} ${ReSolve_DIR}/share/resolve/cmake
     ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
     REQUIRED)

   # Build your executable
   add_executable(my_app my_app.cpp)
   target_link_libraries(my_app PRIVATE ReSolve::ReSolve)


Test and Deploy
---------------

ReSolve as a library is tested every merge request via Gitlab pipelines
that execute various library tests including a test of ReSolve being
consumed as package within an external project as mentioned in `Using
ReSolve in your own
Project <#to-use-the-resolve-library-in-your-own-project>`__

To test your own install of ReSolve simply cd into your ReSolve build
directory and run

.. code:: shell

   $ make test

or

.. code:: shell

   $ ctest

Below is an example of what a functional ReSolve build will ouput on
Passing tests

.. code:: text

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

Important Notes
---------------

You can find default Cmake Configurations in the CMakePresets.json file,
which allows for easy switching between different CMake Configs. To
create your own CMake Configuration we encourage you to utlize a
CmakeUserPresets.json file. To learn more about cmake-presets please
checkout the cmake
`docs <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`__.

For example if you wanted to build and install ReSolve on a High
Performance Computing Cluster such as PNNL’s Deception or ORNL’s Ascent
we encourage you to utilize our cluster preset. Using this preset will
set CMAKE_INSTALL_PREFIX to an install folder. To use this preset simply
call the preset flag in the cmake build step.

.. code:: shell

   cmake -B build --preset cluster

Advanced Builds
====================

CMake Build System
-------------------

Our ``cmake`` folder contains some basic CMake modules that help manage resolve:

* ``cmake/FindKLU.cmake``: Our custom find module for KLU that we maintain
* ``cmake/ReSolveConfig.cmake.in``: Our custom config file that is used to generate the ``ReSolveConfig.cmake`` file that is installed with Re::Solve
* ``cmake/ReSolveFindCudaLibraries.cmake``: Our custom find module for CUDA libraries that we maintain to link in subset of cuda needed
* ``cmake/ReSolveFindHipLibraries.cmake``: Our custom find module for HIP/ROCm libraries that we maintain to link in subset of hip needed

Apart from that check out our main ``CMakeLists.txt`` file for our remaining build configuration.

We also export under the ``ReSolve::`` namespace in our installed CMake configuration for use with ``find_package`` as documented in our main ``README.md``.

Spack Package
---------------

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

Getting Help
------------

For any questions or to report a bug please submit a `GitHub
issue <https://github.com/ORNL/ReSolve/issues>`__.


.. |runResolve| replace:: ``runResolve``
.. _runResolve: https://github.com/ORNL/ReSolve/blob/develop/runResolve

Re::Solve General API Description 
=====================================

Solver Subroutines
------------------

For direct solvers, the following functions are provided and must be called in this order:

* ``analyze()`` - Performs symbolic factorization, currently only with KLU on the CPU.
* ``factorize()`` - Performs numeric factorization, given a symbolic factorization.
* ``refactorizationSetup()`` - Prepares for a new numeric factorization, given a new matrix with the same sparsity structure (required only if using ``refactorize()``).
* ``solve()`` - Solves the linear system, given a numeric factorization.
* ``refactorize()`` - Reuses the symbolic factorization to perform a new numeric factorization, given a new matrix.

Subsequent systems only require calls to ``refactorize()`` and ``solve()`` if the sparsity structure hasn't changed.

Note some examples do not reuse the factorization from the 0-th iteration because the numeric sparsity structure changes and that's how the matrices are stored. In a practical application, the sparsity structure is likely to remain the same. If it does not, it is the user's responsibility to reallocate memory and call ``analyze()`` and ``factorize()`` again. The user should enforce that symbolic nonzeros are stored explicitly, even if they are numerically zero.

Function Requirements
---------------------

Functions must be used as in the examples and tests:

* Workspaces are required for GPU solvers. The generic workspace_type is required for backend agnostic code.
* Handles are created with the ``initializeHandles()`` function and destroyed with the default destructor.
* Allocate memory first, preferably with the ``ReSolve::MatrixHandler`` or ``ReSolve::VectorHandler`` classes.
* Memory must be allocated before attempting to copy to it.
* Solver output variables must be allocated before passing to the solver.
* Use setup functions to initialize objects, where available. Do not call them more than once.
* Call the set or factor functions before the corresponding reset or refactor functions.  
* Deallocate memory at the end. 

Functions in ReSolve that take pointers as arguments will have the following requirements:

* Pointers must be valid and point to allocated memory, unless the function's use is to allocate memory.
* Pointers that are not marked as const must point to memory that is writable.
* Pointers that are marked as const must point to memory that is readable.

Memory Synchronization
----------------------

* For CPU solvers, memory is always in sync.
* For GPU solvers, the user must synchronize memory manually. 
* Manually call ``setUpdated()`` if you modify the contents of a ``ReSolve::MatrixHandler`` or ``ReSolve::VectorHandler`` object without using the object's member functions.
* Manually call ``syncData(memspace)`` when you modify the other memory space and want them to match. Both memory spaces must be allocated.
