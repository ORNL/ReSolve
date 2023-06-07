#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

# Build example with KLU factorization and KLU refactorization
add_executable(klu_klu.exe r_KLU_KLU.cpp)
target_include_directories(klu_klu.exe PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_link_libraries(klu_klu.exe PRIVATE ReSolve)

# Build example with KLU factorization and GLU refactorization
add_executable(klu_glu.exe r_KLU_GLU.cpp)
target_include_directories(klu_glu.exe PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_link_libraries(klu_glu.exe PRIVATE ReSolve)

# Build example with KLU factorization and Rf refactorization
add_executable(klu_rf.exe r_KLU_rf.cpp)
target_include_directories(klu_rf.exe PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_link_libraries(klu_rf.exe PRIVATE ReSolve)

# Build example with KLU factorization, Rf refactorization, and FGMRES iterative refinement
add_executable(klu_rf_fgmres.exe r_KLU_rf_FGMRES.cpp)
target_include_directories(klu_rf_fgmres.exe PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_link_libraries(klu_rf_fgmres.exe PRIVATE ReSolve)
