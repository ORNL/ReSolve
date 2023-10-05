#[[

@brief ReSolve library root

@author Slaven Peles <peless@ornl.gov>

]]

cmake_minimum_required(VERSION 3.20)

# Adds version settings and set variable CMAKE_PROJECT_VERSION
project(ReSolve VERSION "0.1.0")

set(CMAKE_CXX_STANDARD 11)

set(PACKAGE_NAME  "ReSolve")
set(PACKAGE_TARNAME "resolve")

# Prohibit in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source build prohibited.")
endif()

option(RESOLVE_TEST_WITH_BSUB "Use `jsrun` instead of `mpirun` commands when running tests" OFF)
option(RESOLVE_USE_KLU "Use KLU, AMD and COLAMD libraries from SuiteSparse" ON)
set(RESOLVE_CTEST_OUTPUT_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Directory where CTest outputs are saved")

set(CMAKE_MACOSX_RPATH 1)
# set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath
# use, i.e. don't skip the full RPATH for the build tree
#set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
#list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# Add CMake sources from `cmake` dir
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Including clang-format cmake files to do automatic checking of formating
# TODO: Set up clang-format
#include(./cmake/clang-format)

# Configure CUDA
include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)


if (RESOLVE_USE_KLU)
  include(FindKLU)
  if(NOT KLU_LIBRARY)
    message(STATUS "Cannot find KLU, disabling SuiteSparse module ...")
    set(RESOLVE_USE_KLU OFF CACHE BOOL "Build without SuiteSparse AMD module." FORCE)
  endif()
else()
  message(STATUS "Not using SuiteSparse KLU")
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "Selects CUDA architectures")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")


# Link in required cuda dependencies
#find_package(CUDAToolkit REQUIRED)

include(ReSolveFindCudaLibraries)

include_directories(${CMAKE_SOURCE_DIR})

# Enable testing
enable_testing()

# Add directory with library code
add_subdirectory(resolve)

# create package version file
include(CMakePackageConfigHelpers)

# Creates a version file for the package
write_basic_package_version_file(ReSolveConfigVersion.cmake
                                 VERSION ${CMAKE_PROJECT_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

# Generate install rules for targets
install(EXPORT ReSolveTargets
        FILE ReSolveTargets.cmake
        NAMESPACE ReSolve::
        DESTINATION share/resolve/cmake
)


# Creates a config file
configure_package_config_file(./cmake/ReSolveConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/ReSolveConfig.cmake
                              INSTALL_DESTINATION share/resolve/cmake)

# Generates install rules for cmake config files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ReSolveConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ReSolveConfigVersion.cmake"
        DESTINATION share/resolve/cmake)
    
# Add examples
add_subdirectory(examples)

# Add tests
add_subdirectory(tests)
