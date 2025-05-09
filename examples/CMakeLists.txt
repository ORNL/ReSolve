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

  # Build a CPU example with a configurable system solver
  add_executable(system.exe r_SysSolver.cpp)
  target_link_libraries(system.exe PRIVATE ReSolve)

  # Build an example with a configurable and portable system solver
  add_executable(sysRefactor.exe sysRefactor.cpp)
  target_link_libraries(sysRefactor.exe PRIVATE ReSolve)

  if(RESOLVE_USE_GPU)
    # Build an example with refactorization on GPU
    add_executable(gpuRefactor.exe gpuRefactor.cpp)
    target_link_libraries(gpuRefactor.exe PRIVATE ReSolve)
  endif(RESOLVE_USE_GPU)

  # Create KLU+CUDA examples
  if(RESOLVE_USE_CUDA)

    # Build example with KLU factorization and GLU refactorization
    add_executable(gluRefactor.exe gluRefactor.cpp)
    target_link_libraries(gluRefactor.exe PRIVATE ReSolve)

    # Build example with a configurable system solver
    add_executable(system_cuda.exe r_SysSolverCuda.cpp)
    target_link_libraries(system_cuda.exe PRIVATE ReSolve)

  endif(RESOLVE_USE_CUDA)

  # Create KLU+HIP examples
  if(RESOLVE_USE_HIP)

    # Build example with a configurable system solver
    add_executable(system_hip.exe r_SysSolverHip.cpp)
    target_link_libraries(system_hip.exe PRIVATE ReSolve)

    # Build example with a configurable system solver
    add_executable(system_hip_fgmres.exe r_SysSolverHipRefine.cpp)
    target_link_libraries(system_hip_fgmres.exe PRIVATE ReSolve)
  endif(RESOLVE_USE_HIP)

endif(RESOLVE_USE_KLU)


set(installable_executables "")

# Install all examples in bin directory
list(APPEND installable_executables  rand_gmres.exe)

if(RESOLVE_USE_KLU)
  list(APPEND installable_executables kluFactor.exe
                                      kluRefactor.exe
                                      sysRefactor.exe
                                      system.exe)

  if(RESOLVE_USE_GPU)
    list(APPEND installable_executables gpuRefactor.exe)
  endif(RESOLVE_USE_GPU)

  if(RESOLVE_USE_CUDA)
    list(APPEND installable_executables gluRefactor.exe)
  endif(RESOLVE_USE_CUDA)

  if(RESOLVE_USE_HIP)
    list(APPEND installable_executables klu_rocsolverrf_check_redo.exe)
  endif(RESOLVE_USE_HIP)
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
    set(RESOLVE_CONSUMER_APP "testRefactor.cpp")
  elseif(RESOLVE_USE_HIP)
    set(RESOLVE_CONSUMER_APP "testRefactor.cpp")
  else()
    set(RESOLVE_CONSUMER_APP "testKlu.cpp")
  endif()
else(RESOLVE_USE_KLU)
  set(RESOLVE_CONSUMER_APP "testSysGmres.cpp")
endif(RESOLVE_USE_KLU)

# Install directory with example on how to consume ReSolve
install(DIRECTORY resolve_consumer DESTINATION share/examples)
install(FILES ${PROJECT_SOURCE_DIR}/tests/functionality/${RESOLVE_CONSUMER_APP} DESTINATION share/examples/resolve_consumer RENAME consumer.cpp)
install(FILES ${PROJECT_SOURCE_DIR}/tests/functionality/TestHelper.hpp DESTINATION share/examples/resolve_consumer)

# Shell script argumets:
#    1. Path to where resolve is installed.
#    2. Path to data directory
add_custom_target(test_install COMMAND ${CONSUMER_PATH}/test.sh  ${CMAKE_INSTALL_PREFIX} ${PROJECT_SOURCE_DIR}/tests/functionality/)
