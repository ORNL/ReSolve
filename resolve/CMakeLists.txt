#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

set(ReSolve_SRC
    MatrixHandler.cpp
    VectorHandler.cpp
    MatrixIO.cpp
    LinSolver.cpp
    LinSolverDirectKLU.cpp
)


set(ReSolve_SRC_CUDA
    Matrix.cpp
    Vector.cpp
    LinAlgWorkspace.cpp
    LinSolverDirectCuSolverRf.cpp
    LinSolverDirectCuSolverGLU.cpp
    LinSolverIterativeFGMRES.cpp
    cudaKernels.cu
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
    Matrix.hpp
    MatrixIO.hpp
    MatrixHandler.hpp
    RefactorizationSolver.hpp
    SystemSolver.hpp
    Vector.hpp
    VectorHandler.hpp
)

set_source_files_properties(${ReSolve_SRC_CUDA} PROPERTIES LANGUAGE CUDA)

# Build shared library ReSolve
add_library(resolve_tpl INTERFACE)
target_link_libraries(resolve_tpl INTERFACE resolve_cuda KLU)

install(TARGETS resolve_tpl EXPORT ReSolveTargets)

add_library(ReSolve SHARED ${ReSolve_SRC} ${ReSolve_SRC_CUDA})

target_include_directories(ReSolve INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# TODO: Make this PRIVATE dependency (requires refactoring ReSolve code)
target_link_libraries(ReSolve PUBLIC resolve_tpl)

install(TARGETS ReSolve
        EXPORT ReSolveTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
# install include headers
install(FILES ${ReSolve_HEADER_INSTALL} DESTINATION include/resolve)

