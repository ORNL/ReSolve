#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

add_subdirectory(vector)
add_subdirectory(matrix)
add_subdirectory(utilities)

set(ReSolve_SRC
    # VectorHandler.cpp
    LinSolver.cpp
    LinSolverDirectKLU.cpp
)


set(ReSolve_SRC_CUDA
    # Vector.cpp
    LinAlgWorkspace.cpp
    LinSolverDirectCuSolverRf.cpp
    LinSolverDirectCuSolverGLU.cpp
    LinSolverIterativeFGMRES.cpp
    cudaKernels.cu
    GramSchmidt.cpp
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
)

set_source_files_properties(${ReSolve_SRC_CUDA} PROPERTIES LANGUAGE CUDA)

# Build shared library ReSolve
add_library(resolve_tpl INTERFACE)
target_link_libraries(resolve_tpl INTERFACE resolve_cuda KLU)

install(TARGETS resolve_matrix resolve_vector resolve_logger resolve_tpl EXPORT ReSolveTargets)

add_library(ReSolve SHARED ${ReSolve_SRC} ${ReSolve_SRC_CUDA})

target_include_directories(ReSolve INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# TODO: Make this PRIVATE dependency (requires refactoring ReSolve code)
target_link_libraries(ReSolve PUBLIC resolve_matrix resolve_vector resolve_logger resolve_tpl)

install(TARGETS ReSolve
        EXPORT ReSolveTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
# install include headers
install(FILES ${ReSolve_HEADER_INSTALL} DESTINATION include/resolve)

