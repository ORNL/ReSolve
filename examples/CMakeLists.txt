#[[

@brief Build ReSolve examples

@author Slaven Peles <peless@ornl.gov>

]]

# Build example with KLU factorization and KLU refactorization
add_executable(klu_klu.exe r_KLU_KLU.cpp)
target_link_libraries(klu_klu.exe PRIVATE ReSolve)

# Read ONE matrix,solve with KLU
add_executable(klu_klu_standalone.exe r_KLU_KLU_standalone.cpp)
target_link_libraries(klu_klu_standalone.exe PRIVATE ReSolve)

# Create CUDA examples
if(RESOLVE_USE_CUDA)
        
  # Build example with KLU factorization and GLU refactorization
  add_executable(klu_glu.exe r_KLU_GLU.cpp)
  target_link_libraries(klu_glu.exe PRIVATE ReSolve)

  # Build example with KLU factorization and Rf refactorization
  add_executable(klu_rf.exe r_KLU_rf.cpp)
  target_link_libraries(klu_rf.exe PRIVATE ReSolve)

  # Build example with KLU factorization, Rf refactorization, and FGMRES iterative refinement
  add_executable(klu_rf_fgmres.exe r_KLU_rf_FGMRES.cpp)
  target_link_libraries(klu_rf_fgmres.exe PRIVATE ReSolve)

  # Build example where matrix is factorized once, refactorized once and then the preconditioner is REUSED
  add_executable(klu_rf_fgmres_reuse_refactorization.exe r_KLU_rf_FGMRES_reuse_factorization.cpp)
  target_link_libraries(klu_rf_fgmres_reuse_refactorization.exe PRIVATE ReSolve)

  # Build example where matrix data is updated 
  add_executable(klu_glu_values_update.exe r_KLU_GLU_matrix_values_update.cpp)
  target_link_libraries(klu_glu_values_update.exe PRIVATE ReSolve)

endif(RESOLVE_USE_CUDA)



# Install all examples in bin directory
set(installable_executables klu_klu.exe klu_klu_standalone.exe)

if(RESOLVE_USE_CUDA)
  set(installable_executables ${installable_executables} klu_glu.exe klu_rf.exe klu_rf_fgmres.exe klu_glu_values_update.exe)        
endif(RESOLVE_USE_CUDA)

install(TARGETS ${installable_executables} 
        RUNTIME DESTINATION bin)

# Path where the consumer test code will be installed
set(CONSUMER_PATH ${CMAKE_INSTALL_PREFIX}/share/examples)

# Make the resolve consumer test script exectuable
install(PROGRAMS test.sh DESTINATION ${CONSUMER_PATH})

# Install directory with example on how to consume ReSolve
install(DIRECTORY resolve_consumer DESTINATION share/examples)
install(FILES ${PROJECT_SOURCE_DIR}/tests/functionality/testKLU_Rf_FGMRES.cpp DESTINATION share/examples/resolve_consumer)

# Shell script argumets: 
#    1. Path to where resolve is installed.
#    2. Path to data directory
add_custom_target(test_install COMMAND ${CONSUMER_PATH}/test.sh  ${CMAKE_INSTALL_PREFIX} ${PROJECT_SOURCE_DIR}/tests/functionality/)
