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

set_source_files_properties(${ReSolve_SRC_CUDA} PROPERTIES LANGUAGE CUDA)

# Build shared library ReSolve
add_library(ReSolve SHARED ${ReSolve_SRC} ${ReSolve_SRC_CUDA})
target_include_directories(ReSolve INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(ReSolve PRIVATE KLU resolve_cuda)
