#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

add_subdirectory(utilities)

set(ReSolve_SRC
    LinSolver.cpp
    LinSolverDirectKLU.cpp
    LinAlgWorkspace.cpp
    LinSolverIterativeFGMRES.cpp
    GramSchmidt.cpp
    LinSolverDirectCuSolverGLU.cpp
    LinSolverDirectCuSolverRf.cpp
)


set(ReSolve_SRC_CUDA
    cudaKernels.cu
    memoryUtils.cu
)

set(ReSolve_HEADER_INSTALL
    Common.hpp
    cudaKernels.h
    cusolver_defs.hpp
    LinAlgWorkspace.hpp
    LinSolver.hpp
    LinSolverDirectCuSolverGLU.hpp
    LinSolverDirectCuSolverRf.hpp
    LinSolverDirectKLU.hpp
    LinSolverIterativeFGMRES.hpp
    RefactorizationSolver.hpp
    SystemSolver.hpp
    GramSchmidt.hpp
    memoryUtils.hpp
)

set_source_files_properties(${ReSolve_SRC_CUDA} PROPERTIES LANGUAGE CUDA)

# First create CUDA backend (this should really be CUDA _API_ backend, separate backend will be needed for CUDA SDK)
add_library(resolve_backend_cuda SHARED ${ReSolve_SRC_CUDA})
target_link_libraries(resolve_backend_cuda PUBLIC resolve_cuda)
target_include_directories(resolve_backend_cuda INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Next build vector and matrix objects that may use this backend.
add_subdirectory(vector)
add_subdirectory(matrix)


# Build shared library ReSolve
add_library(resolve_tpl INTERFACE)
target_link_libraries(resolve_tpl INTERFACE resolve_cuda KLU)

install(TARGETS resolve_matrix resolve_vector resolve_backend_cuda resolve_logger resolve_tpl EXPORT ReSolveTargets)

add_library(ReSolve SHARED ${ReSolve_SRC})

target_include_directories(ReSolve INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# TODO: Make this PRIVATE dependency (requires refactoring ReSolve code)
target_link_libraries(ReSolve PUBLIC resolve_matrix resolve_vector resolve_backend_cuda resolve_logger resolve_tpl)

install(TARGETS ReSolve
        EXPORT ReSolveTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
# install include headers
install(FILES ${ReSolve_HEADER_INSTALL} DESTINATION include/resolve)

