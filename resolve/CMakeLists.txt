#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

add_subdirectory(utilities)

# C++ files
set(ReSolve_SRC
    LinSolver.cpp
    LinSolverDirectKLU.cpp
)

# C++ code that links to CUDA SDK libraries
set(ReSolve_CUDASDK_SRC
    LinSolverIterativeFGMRES.cpp
    GramSchmidt.cpp
    LinSolverDirectCuSolverGLU.cpp
    LinSolverDirectCuSolverRf.cpp
)

# Header files to be installed
set(ReSolve_HEADER_INSTALL
    Common.hpp
    cusolver_defs.hpp
    LinSolver.hpp
    LinSolverDirectCuSolverGLU.hpp
    LinSolverDirectCuSolverRf.hpp
    LinSolverDirectKLU.hpp
    LinSolverIterativeFGMRES.hpp
    RefactorizationSolver.hpp
    SystemSolver.hpp
    GramSchmidt.hpp
    MemoryUtils.hpp
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
endif(RESOLVE_USE_KLU)

set(ReSolve_Targets_List
    resolve_matrix
    resolve_vector
    resolve_logger
    resolve_tpl
    resolve_workspace
)

# If CUDA support is enabled add CUDA SDK specific code and dependencies
if(RESOLVE_USE_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(resolve_tpl INTERFACE resolve_cuda)
  set(ReSolve_SRC ${ReSolve_SRC} ${ReSolve_CUDASDK_SRC})
  set(ReSolve_Targets_List ${ReSolve_Targets_List} resolve_backend_cuda)
endif()

# If HIP support is enabled add HIP SDK specific code and dependencies
if(RESOLVE_USE_HIP)
  add_subdirectory(hip)
  target_link_libraries(resolve_tpl INTERFACE resolve_hip)
  set(ReSolve_Targets_List ${ReSolve_Targets_List} resolve_backend_hip)
endif()

# If no GPU support is enabled, link to dummy device backend
if(NOT RESOLVE_USE_GPU)
  add_subdirectory(cpu)
  set(ReSolve_Targets_List ${ReSolve_Targets_List} resolve_backend_cpu)
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

# Install targets
install(TARGETS ReSolve
        EXPORT ReSolveTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# install include headers
install(FILES ${ReSolve_HEADER_INSTALL} DESTINATION include/resolve)

