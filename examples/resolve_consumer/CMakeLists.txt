cmake_minimum_required(VERSION 3.20)
project(klu_klu.exe LANGUAGES CXX)

# path to ReSolve CMake config file (optinal if ReSolve is installed globally)
find_package(ReSolve CONFIG 
  PATHS ${ReSolve_DIR} ${ReSolve_DIR}/share/resolve/cmake
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  REQUIRED)

# Build example with KLU factorization and KLU refactorization
add_executable(klu_rf_fgmres.exe r_KLU_rf_FGMRES.cpp)
target_link_libraries(klu_rf_fgmres.exe PRIVATE ReSolve::ReSolve)