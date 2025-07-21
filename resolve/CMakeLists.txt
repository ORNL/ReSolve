#[[

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

set(ReSolve_KLU_SRC LinSolverDirectKLU.cpp)

set(ReSolve_LUSOL_SRC LinSolverDirectLUSOL.cpp)


# C++ code that links to CUDA SDK libraries
set(ReSolve_CUDASDK_SRC
    LinSolverDirectCuSolverGLU.cpp LinSolverDirectCuSolverRf.cpp
    LinSolverDirectCuSparseILU0.cpp
)

# C++ code that links to ROCm libraries
set(ReSolve_ROCM_SRC LinSolverDirectRocSolverRf.cpp
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
    MemoryUtils.hpp
)

set(ReSolve_KLU_HEADER_INSTALL LinSolverDirectKLU.hpp)

set(ReSolve_LUSOL_HEADER_INSTALL LinSolverDirectLUSOL.hpp)

set(ReSolve_CUDA_HEADER_INSTALL
    LinSolverDirectCuSolverGLU.hpp LinSolverDirectCuSolverRf.hpp
    LinSolverDirectCuSparseILU0.hpp
)

set(ReSolve_ROCM_HEADER_INSTALL LinSolverDirectRocSolverRf.hpp
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

set(ReSolve_Targets_List resolve_matrix resolve_vector resolve_random
                         resolve_logger resolve_tpl resolve_workspace
)

set(ReSolve_Object_List resolve_options resolve_version)

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
  target_link_libraries(
    resolve_asan INTERFACE "-fsanitize=address -fno-omit-frame-pointer"
  )
  target_link_libraries(resolve_tpl INTERFACE resolve_asan)
  list(APPEND ReSolve_Targets_List resolve_asan)
endif()

# Undefined behavior sanitizer target
if(RESOLVE_USE_UBSAN)
  add_library(resolve_ubsan INTERFACE)
  target_compile_options(resolve_ubsan INTERFACE "-fsanitize=undefined")
  target_link_libraries(
    resolve_ubsan INTERFACE "-fsanitize=undefined -fno-omit-frame-pointer"
  )
  target_link_libraries(resolve_tpl INTERFACE resolve_ubsan)
  list(APPEND ReSolve_Targets_List resolve_ubsan)
endif()

# Add HyKKT solver
if(RESOLVE_USE_KLU)
  add_subdirectory(hykkt)
endif()
#list(APPEND ReSolve_Targets_List resolve_hykkt)

# Set installable targets
install(TARGETS ${ReSolve_Targets_List} EXPORT ReSolveTargets)

# Create ReSolve library
add_library(ReSolve SHARED ${ReSolve_SRC})

target_include_directories(
  ReSolve INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ReSolve PUBLIC ${ReSolve_Targets_List})
target_link_libraries(ReSolve PRIVATE ${ReSolve_Object_List})

if(RESOLVE_USE_PROFILING)
  if(NOT RESOLVE_USE_GPU)
    # Noting to do for profiling on the host for now.
    message(
      NOTICE
      "Profiling support enabled, but Re::Solve does not create tracer annotations for host code."
    )
    message(NOTICE "This profiling support option will have no effect.")
  endif()
endif(RESOLVE_USE_PROFILING)

# Install targets
install(
  TARGETS ReSolve
  EXPORT ReSolveTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

# install include headers
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/resolve/ # source directory
  DESTINATION include/resolve # target directory
  FILES_MATCHING # install only matched files
  PATTERN "*.hpp" # select header files
)
