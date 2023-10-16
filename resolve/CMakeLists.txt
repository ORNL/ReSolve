#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

add_subdirectory(utilities)

set(ReSolve_SRC
    LinSolver.cpp
    LinSolverDirectKLU.cpp
    LinSolverIterativeFGMRES.cpp
    GramSchmidt.cpp
    LinSolverDirectCuSolverGLU.cpp
    LinSolverDirectCuSolverRf.cpp
)


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

# If GPU support is not enabled, add dummy device backend
if(NOT RESOLVE_USE_GPU)
    add_subdirectory(cpu)
endif()

# If CUDA support is enabled, create CUDA backend
# (this should really be CUDA _API_ backend, separate backend will be needed for CUDA SDK)
if(RESOLVE_USE_CUDA)
    add_subdirectory(cuda)
endif()

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

if(RESOLVE_USE_CUDA)
    target_link_libraries(resolve_tpl INTERFACE resolve_cuda)
endif(RESOLVE_USE_CUDA)


set(ReSolve_Targets_List
    resolve_matrix
    resolve_vector
    resolve_logger
    resolve_tpl
    resolve_workspace
)

if(RESOLVE_USE_CUDA)
  set(ReSolve_Targets_List ${ReSolve_Targets_List} resolve_backend_cuda)
endif()

if(NOT RESOLVE_USE_GPU)
    set(ReSolve_Targets_List ${ReSolve_Targets_List} resolve_backend_cpu)
endif()


install(TARGETS ${ReSolve_Targets_List} EXPORT ReSolveTargets)

add_library(ReSolve SHARED ${ReSolve_SRC})

target_include_directories(ReSolve INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# TODO: Make this PRIVATE dependency (requires refactoring ReSolve code)
target_link_libraries(ReSolve PUBLIC ${ReSolve_Targets_List})

install(TARGETS ReSolve
        EXPORT ReSolveTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
# install include headers
install(FILES ${ReSolve_HEADER_INSTALL} DESTINATION include/resolve)

