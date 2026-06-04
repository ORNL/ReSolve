#[[

@brief Re::Solve library root

@author Slaven Peles <peless@ornl.gov>

]]

cmake_minimum_required(VERSION 3.22)

# Adds version settings and set variable CMAKE_PROJECT_VERSION
project(ReSolve VERSION "0.99.2")

set(CMAKE_CXX_STANDARD 11)

set(PACKAGE_NAME "ReSolve")
set(PACKAGE_TARNAME "resolve")

# Prohibit in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source build prohibited.")
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "/Wall")
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wpedantic")
endif()

option(RESOLVE_TEST_WITH_BSUB
       "Use `jsrun` instead of `mpirun` commands when running tests" OFF
)
option(RESOLVE_USE_KLU "Use KLU, AMD and COLAMD libraries from SuiteSparse" OFF)
option(RESOLVE_USE_LUSOL "Build the LUSOL library" OFF)
option(RESOLVE_USE_CUDA "Use CUDA language and SDK" OFF)
option(RESOLVE_USE_HIP "Use HIP language and ROCm library" OFF)
option(RESOLVE_USE_PROFILING "Set profiling tracers in the code" OFF)

option(RESOLVE_USE_GPU "Use GPU device for computations" OFF)
mark_as_advanced(FORCE RESOLVE_USE_GPU)

option(RESOLVE_USE_ASAN "Enable the address sanitizer" OFF)
option(RESOLVE_USE_UBSAN "Enable the undefined behavior sanitizer" OFF)
option(RESOLVE_USE_DOXYGEN "Use Doxygen to generate Re::Solve documentation"
       OFF
)
set(RESOLVE_CTEST_OUTPUT_DIR
    ${PROJECT_BINARY_DIR}
    CACHE PATH "Directory where CTest outputs are saved"
)

# update this if more fortran code is added. this should be good enough for now
# though
if(RESOLVE_USE_LUSOL)
  enable_language(Fortran)
endif()

if(RESOLVE_USE_CUDA)
  set(RESOLVE_USE_GPU
      ON
      CACHE BOOL "Using CUDA GPU!" FORCE
  )
endif()

if(RESOLVE_USE_HIP)
  set(RESOLVE_USE_GPU
      ON
      CACHE BOOL "Using HIP GPU!" FORCE
  )
endif()

# MacOS specific things
set(CMAKE_MACOSX_RPATH 1)

# Install with RPATH but do not build with it
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# Add CMake sources from `cmake` dir
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Including clang-format cmake files to do automatic checking of formating TODO:
# Set up clang-format include(./cmake/clang-format)

# This will create target `doxygen` for building documentation locally. For now,
# this target is intended only for developers who want to test different
# documentation configurations. To have Doxygen configuration changed at
# readthedocs, update `docs/doxygen/Doxyfile.in` accordingly.
if(RESOLVE_USE_DOXYGEN)
  include(ReSolveFindDoxygen)
endif()

if(RESOLVE_USE_KLU)
  include(FindSuiteSparse)
  if(NOT SUITESPARSE_LIBRARY)
    message(STATUS "Cannot find KLU, disabling SuiteSparse module ...")
    set(RESOLVE_USE_KLU
        OFF
        CACHE BOOL "Build without SuiteSparse module." FORCE
    )
  endif()
else()
  message(STATUS "Not using SuiteSparse KLU")
endif()

include(CheckLanguage)

# Configure CUDA
if(RESOLVE_USE_CUDA)
  enable_language(CUDA)
  check_language(CUDA)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES
        60
        CACHE STRING "Selects CUDA architectures"
    )
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

  include(ReSolveFindCudaLibraries)
else()
  message(STATUS "Not using CUDA")
endif()

if(RESOLVE_USE_HIP)
  enable_language(HIP)
  check_language(HIP)
  include(ReSolveFindHipLibraries)
else()
  message(STATUS "Not using HIP")
endif(RESOLVE_USE_HIP)

# The binary dir is already a global include directory
configure_file(
  ${CMAKE_SOURCE_DIR}/resolve/resolve_defs.hpp.in
  ${CMAKE_BINARY_DIR}/resolve/resolve_defs.hpp
)
install(FILES ${CMAKE_BINARY_DIR}/resolve/resolve_defs.hpp
        DESTINATION include/resolve
)

# Enable testing
enable_testing()

# Add directory with library code
add_subdirectory(resolve)

# create package version file
include(CMakePackageConfigHelpers)

# Creates a version file for the package
write_basic_package_version_file(
  ReSolveConfigVersion.cmake
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Generate install rules for targets
install(
  EXPORT ReSolveTargets
  FILE ReSolveTargets.cmake
  NAMESPACE ReSolve::
  DESTINATION share/resolve/cmake
)

# Creates a config file
configure_package_config_file(
  ./cmake/ReSolveConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ReSolveConfig.cmake
  INSTALL_DESTINATION share/resolve/cmake
)

# Generates install rules for cmake config files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ReSolveConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/ReSolveConfigVersion.cmake"
        DESTINATION share/resolve/cmake
)

# Add usage examples
add_subdirectory(examples)

# Add tests
set(RESOLVE_CTEST_OUTPUT_DIR
    ${PROJECT_BINARY_DIR}
    CACHE PATH "Directory where CTest outputs are saved"
)
add_subdirectory(tests)
