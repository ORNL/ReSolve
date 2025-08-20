#[[
<<<<<<< HEAD

@brief Build Re::Solve library

@author Slaven Peles <peless@ornl.gov>

]]

add_subdirectory(utilities)

# C++ files
set(ReSolve_SRC
    Common.hpp
    LinSolver.cpp
    LinSolverDirect.cpp
    LinSolverIterative.cpp
    GramSchmidt.cpp
    LinSolverIterativeFGMRES.cpp
    LinSolverDirectCpuILU0.cpp
    LinSolverIterativeRandFGMRES.cpp
    LinSolverDirectSerialILU0.cpp
    SystemSolver.cpp
)

set(ReSolve_KLU_SRC
    LinSolverDirectKLU.cpp
)

set(ReSolve_LUSOL_SRC
    LinSolverDirectLUSOL.cpp
)

# C++ code that links to CUDA SDK libraries
set(ReSolve_CUDASDK_SRC
    LinSolverDirectCuSolverGLU.cpp
    LinSolverDirectCuSolverRf.cpp
    LinSolverDirectCuSparseILU0.cpp
)

# C++ code that links to ROCm libraries
set(ReSolve_ROCM_SRC
    LinSolverDirectRocSolverRf.cpp
    LinSolverDirectRocSparseILU0.cpp
)

# Header files to be installed
set(ReSolve_HEADER_INSTALL
    Common.hpp
    cusolver_defs.hpp
    LinSolver.hpp
    LinSolverDirect.hpp
    LinSolverIterative.hpp
    LinSolverIterativeFGMRES.hpp
    LinSolverDirectCpuILU0.hpp
    SystemSolver.hpp
    GramSchmidt.hpp
    MemoryUtils.hpp)

set(ReSolve_KLU_HEADER_INSTALL
    LinSolverDirectKLU.hpp
)

set(ReSolve_LUSOL_HEADER_INSTALL
    LinSolverDirectLUSOL.hpp
)

set(ReSolve_CUDA_HEADER_INSTALL
    LinSolverDirectCuSolverGLU.hpp
    LinSolverDirectCuSolverRf.hpp
    LinSolverDirectCuSparseILU0.hpp
)

set(ReSolve_ROCM_HEADER_INSTALL
    LinSolverDirectRocSolverRf.hpp
    LinSolverDirectRocSparseILU0.hpp
)

# Now, build workspaces
add_subdirectory(workspace)

# Next build vector and matrix objects that may use this backend.
add_subdirectory(vector)
add_subdirectory(matrix)
add_subdirectory(random)

# Build shared library ReSolve
add_library(resolve_tpl INTERFACE)

if(RESOLVE_USE_KLU)
    target_link_libraries(resolve_tpl INTERFACE ${suitesparse_klu})
    target_include_directories(resolve_tpl INTERFACE ${SUITESPARSE_INCLUDE_DIR})
    list(APPEND ReSolve_SRC ${ReSolve_KLU_SRC})
    list(APPEND ReSolve_HEADER_INSTALL ${ReSolve_KLU_HEADER_INSTALL})
endif()

set(ReSolve_Targets_List
    resolve_matrix
    resolve_vector
    resolve_random
    resolve_logger
    resolve_tpl
    resolve_workspace
)

set(ReSolve_Object_List
    resolve_options
    resolve_version
)

if(RESOLVE_USE_LUSOL)
    add_subdirectory(lusol)
    list(APPEND ReSolve_SRC ${ReSolve_LUSOL_SRC})
    list(APPEND ReSolve_HEADER_INSTALL ${ReSolve_LUSOL_HEADER_INSTALL})
    list(APPEND ReSolve_Targets_List lusol_lib)
endif()

# If no GPU support is enabled, link to dummy device backend
if(NOT RESOLVE_USE_GPU)
    add_subdirectory(cpu)
    list(APPEND ReSolve_Targets_List resolve_backend_cpu)
endif(NOT RESOLVE_USE_GPU)

# If CUDA support is enabled add CUDA SDK specific code and dependencies
if(RESOLVE_USE_CUDA)
    add_subdirectory(cuda)
    target_link_libraries(resolve_tpl INTERFACE resolve_cuda)
    list(APPEND ReSolve_SRC ${ReSolve_CUDASDK_SRC})
    list(APPEND ReSolve_Targets_List resolve_backend_cuda)
    list(APPEND ReSolve_HEADER_INSTALL ${ReSolve_CUDA_HEADER_INSTALL})
endif()

# If HIP support is enabled add HIP SDK specific code and dependencies
if(RESOLVE_USE_HIP)
    add_subdirectory(hip)
    target_link_libraries(resolve_tpl INTERFACE resolve_hip)
    list(APPEND ReSolve_SRC ${ReSolve_ROCM_SRC})
    list(APPEND ReSolve_Targets_List resolve_backend_hip)
    list(APPEND ReSolve_HEADER_INSTALL ${ReSolve_ROCM_HEADER_INSTALL})
endif()

# Address sanitizer target
if(RESOLVE_USE_ASAN)
    add_library(resolve_asan INTERFACE)
    target_compile_options(resolve_asan INTERFACE "-fsanitize=address")
    target_link_libraries(resolve_asan INTERFACE "-fsanitize=address -fno-omit-frame-pointer")
    target_link_libraries(resolve_tpl INTERFACE resolve_asan)
    list(APPEND ReSolve_Targets_List resolve_asan)
endif()

# Undefined behavior sanitizer target
if(RESOLVE_USE_UBSAN)
    add_library(resolve_ubsan INTERFACE)
    target_compile_options(resolve_ubsan INTERFACE "-fsanitize=undefined")
    target_link_libraries(resolve_ubsan INTERFACE "-fsanitize=undefined -fno-omit-frame-pointer")
    target_link_libraries(resolve_tpl INTERFACE resolve_ubsan)
    list(APPEND ReSolve_Targets_List resolve_ubsan)
endif()

# Set installable targets
install(TARGETS ${ReSolve_Targets_List} EXPORT ReSolveTargets)

# Create ReSolve library
add_library(ReSolve SHARED ${ReSolve_SRC})

# Corrected include directories for the ReSolve library
target_include_directories(ReSolve PUBLIC
    # This includes the root of the 'resolve' directory
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>

    # This is the corrected path to the 'utilities/logger' directory
    # It ensures the compiler can find Logger.hpp
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utilities/logger>

    $<INSTALL_INTERFACE:include>
)


target_link_libraries(ReSolve PUBLIC ${ReSolve_Targets_List})
target_link_libraries(ReSolve PRIVATE ${ReSolve_Object_List})

if(RESOLVE_USE_PROFILING)
    if(NOT RESOLVE_USE_GPU)
        # Noting to do for profiling on the host for now.
        message(NOTICE "Profiling support enabled, but Re::Solve does not create tracer annotations for host code.")
        message(NOTICE "This profiling support option will have no effect.")
    endif()
endif(RESOLVE_USE_PROFILING)

# Install targets
install(TARGETS ReSolve
    EXPORT ReSolveTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

# install include headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/resolve/ # source directory
    DESTINATION include/resolve # target directory
    FILES_MATCHING # install only matched files
    PATTERN "*.hpp" # select header files
)
=======
@brief Build Re::Solve examples

@author Slaven Peles <peless@ornl.gov>
]]


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

  # Create KLU+CUDA examples
  if(RESOLVE_USE_CUDA)

    # Build example with KLU factorization and GLU refactorization
    add_executable(hybrid_solver.exe hybrid_solver.cpp)
    target_link_libraries(hybrid_solver.exe PRIVATE ReSolve)

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
>>>>>>> origin/master
