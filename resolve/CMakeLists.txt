#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

add_subdirectory(utilities)

# C++ files
set(ReSolve_SRC
    LinSolver.cpp
    GramSchmidt.cpp
    LinSolverIterativeFGMRES.cpp
)

# Temporary until there is CPU-only option for FGMRES
set(ReSolve_GPU_SRC
    LinSolverIterativeRandFGMRES.cpp
    RandSketchingManager.cpp
    RandSketchingCountSketch.cpp
    RandSketchingFWHT.cpp
)

set(ReSolve_KLU_SRC
  LinSolverDirectKLU.cpp
  SystemSolver.cpp
)

# C++ code that links to CUDA SDK libraries
set(ReSolve_CUDASDK_SRC
    LinSolverDirectCuSolverGLU.cpp
    LinSolverDirectCuSolverRf.cpp
    LinSolverDirectCuSparseILU0.cpp
)

# HIP files
set(ReSolve_ROCM_SRC
  LinSolverDirectRocSolverRf.cpp
  LinSolverDirectRocSparseILU0.cpp
)
# Header files to be installed
set(ReSolve_HEADER_INSTALL
    Common.hpp
    cusolver_defs.hpp
    LinSolver.hpp
    LinSolverIterativeFGMRES.hpp
    RefactorizationSolver.hpp
    SystemSolver.hpp
    GramSchmidt.hpp
    MemoryUtils.hpp)

set(ReSolve_KLU_HEADER_INSTALL
  LinSolverDirectKLU.hpp
)

set(ReSolve_GPU_HEADER_INSTALL
    RandSketchingManager.hpp
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

# Build shared library ReSolve
add_library(resolve_tpl INTERFACE)

if(RESOLVE_USE_KLU)
  target_link_libraries(resolve_tpl INTERFACE KLU)
  list(APPEND ReSolve_SRC ${ReSolve_KLU_SRC})
  list(APPEND ReSolve_HEADER_INSTALL ${ReSolve_KLU_HEADER_INSTALL})
endif()

set(ReSolve_Targets_List
    resolve_matrix
    resolve_vector
    resolve_logger
    resolve_tpl
    resolve_workspace
)

# Temporary until there is CPU-only option for FGMRES
if(RESOLVE_USE_GPU)
  list(APPEND ReSolve_SRC ${ReSolve_GPU_SRC})
  list(APPEND ReSolve_HEADER_INSTALL ${ReSolve_GPU_HEADER_INSTALL})
endif()

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

# If no GPU support is enabled, link to dummy device backend
if(NOT RESOLVE_USE_GPU)
  add_subdirectory(cpu)
  list(APPEND ReSolve_Targets_List resolve_backend_cpu)
endif()

# Set installable targets
install(TARGETS ${ReSolve_Targets_List} EXPORT ReSolveTargets)

# Create ReSolve library
add_library(ReSolve SHARED ${ReSolve_SRC})

target_include_directories(ReSolve INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# TODO: Make this PRIVATE dependency (requires refactoring ReSolve code)
target_link_libraries(ReSolve PUBLIC ${ReSolve_Targets_List})
target_link_libraries(ReSolve PRIVATE resolve_version)

# Install targets
install(TARGETS ReSolve
        EXPORT ReSolveTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# install include headers
install(FILES ${ReSolve_HEADER_INSTALL} DESTINATION include/resolve)

