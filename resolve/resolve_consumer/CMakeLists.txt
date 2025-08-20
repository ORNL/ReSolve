# Example of how to consume Resolve as package via cmake
# Below is a standard example of a CMakeLists.txt file that ultizies Resolve
# See the Readme on how to build and install ReSolve
#----------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

project(resolve_consumer LANGUAGES CXX)

# path to ReSolve CMake config file (optinal if ReSolve is installed globally)
find_package(ReSolve REQUIRED)

# Build example with KLU factorization and KLU refactorization
add_executable(consume.exe consumer.cpp)
target_link_libraries(consume.exe PRIVATE ReSolve::ReSolve)

#------------------------------------------------------------------------------------
# Testing of exported Resolve Configurations
enable_testing()

# RESOLVE_DATA is set in test.sh and is the file path the matrix data files used in the testKLU_Rf_FGMRES
add_test(NAME resolve_consumer COMMAND $<TARGET_FILE:consume.exe> "-d" "${RESOLVE_DATA}" "-i")
