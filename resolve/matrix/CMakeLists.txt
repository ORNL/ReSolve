#[[

@brief Build ReSolve matrix module

@author Slaven Peles <peless@ornl.gov>

]]

# C++ code
set(Matrix_SRC 
    io.cpp
    Sparse.cpp
    Csr.cpp
    Csc.cpp
    Coo.cpp
    MatrixHandler.cpp
    MatrixHandlerCpu.cpp
)

# C++ code that depends on CUDA SDK libraries
set(Matrix_CUDASDK_SRC 
    MatrixHandlerCuda.cpp
)

# Header files to be installed
set(Matrix_HEADER_INSTALL
    io.hpp
    Sparse.hpp
    Coo.hpp
    Csr.hpp
    Csc.hpp
    MatrixHandler.hpp
)

# Add CUDA matrix handler if CUDA support is enabled
if(RESOLVE_USE_CUDA)
  set(Matrix_SRC ${Matrix_SRC} ${Matrix_CUDASDK_SRC})
endif()


# Build shared library ReSolve::matrix
add_library(resolve_matrix SHARED ${Matrix_SRC})
target_link_libraries(resolve_matrix PRIVATE resolve_logger resolve_vector)

# Link to CUDA ReSolve backend if CUDA is support enabled
if (RESOLVE_USE_CUDA)
  target_link_libraries(resolve_matrix PUBLIC resolve_backend_cuda)
endif()

# Link to dummy device backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_matrix PUBLIC resolve_backend_cpu)
endif()


target_include_directories(resolve_matrix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# # TODO: Make this PRIVATE dependency (requires refactoring ReSolve code)
# target_link_libraries(ReSolve PUBLIC resolve_tpl)

# install(TARGETS ReSolve
#         EXPORT ReSolveTargets
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib)
# install include headers
install(FILES ${Matrix_HEADER_INSTALL} DESTINATION include/resolve/matrix)

