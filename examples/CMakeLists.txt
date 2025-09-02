#[[

@brief Build Re::Solve examples

@author Slaven Peles <peless@ornl.gov>

]]

# Build portable randomized GMRES example
add_executable(rand_gmres.exe randGmres.cpp)
target_link_libraries(rand_gmres.exe PRIVATE ReSolve)

if(RESOLVE_USE_KLU)

  # Build example with KLU factorization on CPU
  add_executable(kluFactor.exe kluFactor.cpp)
  target_link_libraries(kluFactor.exe PRIVATE ReSolve)

  # Build example with KLU factorization and KLU refactorization
  add_executable(kluRefactor.exe kluRefactor.cpp)
  target_link_libraries(kluRefactor.exe PRIVATE ReSolve)

  # Build an example with a configurable and portable system solver
  add_executable(sysRefactor.exe sysRefactor.cpp)
  target_link_libraries(sysRefactor.exe PRIVATE ReSolve)

  if(RESOLVE_USE_GPU)
    # Build an example with refactorization on GPU
    add_executable(gpuRefactor.exe gpuRefactor.cpp)
    target_link_libraries(gpuRefactor.exe PRIVATE ReSolve)
  endif(RESOLVE_USE_GPU)

  if(RESOLVE_USE_HIP) #HIP specific example
    add_executable(gpuRefactorSame.exe gpuRefactorSame.cpp)
    target_link_libraries(gpuRefactorSame.exe PRIVATE ReSolve)
  endif(RESOLVE_USE_HIP)

  # Create KLU+CUDA examples
  if(RESOLVE_USE_CUDA)

    # Build example with KLU factorization and GLU refactorization
    add_executable(gluRefactor.exe gluRefactor.cpp)
    target_link_libraries(gluRefactor.exe PRIVATE ReSolve)

  endif(RESOLVE_USE_CUDA)

endif(RESOLVE_USE_KLU)


set(installable_executables "")

# Install all examples in bin directory
list(APPEND installable_executables  rand_gmres.exe)

if(RESOLVE_USE_KLU)
  list(APPEND installable_executables kluFactor.exe
                                      kluRefactor.exe
                                      sysRefactor.exe)

  if(RESOLVE_USE_GPU)
    list(APPEND installable_executables gpuRefactor.exe)
  endif(RESOLVE_USE_GPU)

  if(RESOLVE_USE_CUDA)
    list(APPEND installable_executables gluRefactor.exe)
  endif(RESOLVE_USE_CUDA)

endif(RESOLVE_USE_KLU)

install(TARGETS ${installable_executables}
        RUNTIME DESTINATION bin)

add_subdirectory(experimental)

#
# Consumer example for test_install target
#

# Path where the consumer test code will be installed
set(CONSUMER_PATH ${CMAKE_INSTALL_PREFIX}/share/examples)

# Make the resolve consumer test script exectuable
install(PROGRAMS test.sh DESTINATION ${CONSUMER_PATH})

# Select consumer app
if(RESOLVE_USE_KLU)
  if(RESOLVE_USE_CUDA)
    set(RESOLVE_CONSUMER_APP "testSysRefactor.cpp")
  elseif(RESOLVE_USE_HIP)
    set(RESOLVE_CONSUMER_APP "testSysRefactor.cpp")
  else()
    set(RESOLVE_CONSUMER_APP "testKlu.cpp")
  endif()
else(RESOLVE_USE_KLU)
  set(RESOLVE_CONSUMER_APP "testSysGmres.cpp")
endif(RESOLVE_USE_KLU)

set(test_data_dir ${CMAKE_SOURCE_DIR}/tests/functionality)
set(sym_matrix_path /data/matrix_ACTIVSg200_AC_renumbered_add9_)
set(sym_rhs_path /data/rhs_ACTIVSg200_AC_renumbered_add9_ones_)
set(asym_matrix_path /data/A_asymmetric)
set(asym_rhs_path /data/b_asymmetric)

# Install directory with example on how to consume ReSolve
install(DIRECTORY resolve_consumer DESTINATION share/examples)
install(FILES ${PROJECT_SOURCE_DIR}/tests/functionality/${RESOLVE_CONSUMER_APP} DESTINATION share/examples/resolve_consumer RENAME consumer.cpp)
install(FILES ${PROJECT_SOURCE_DIR}/tests/functionality/TestHelper.hpp DESTINATION share/examples/resolve_consumer)

# Shell script argumets:
#    1. Path to where resolve is installed.
#    2. Path to data directory
add_custom_target(test_install COMMAND ${CONSUMER_PATH}/test.sh  ${CMAKE_INSTALL_PREFIX} ${test_data_dir} ${sym_matrix_path} ${sym_rhs_path}) 
