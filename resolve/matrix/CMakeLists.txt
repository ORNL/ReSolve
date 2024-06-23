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
    utilities.h
)

# C++ code that depends on CUDA SDK libraries
set(Matrix_CUDASDK_SRC
    MatrixHandlerCuda.cpp
)

# and on HIP
set(Matrix_ROCM_SRC
  MatrixHandlerHip.cpp
)

# Header files to be installed
set(Matrix_HEADER_INSTALL
    io.hpp
    Sparse.hpp
    Coo.hpp
    Csr.hpp
    Csc.hpp
    MatrixHandler.hpp
    utilities.cpp
)

# Build shared library ReSolve::matrix
add_library(resolve_matrix SHARED ${Matrix_SRC})
target_link_libraries(resolve_matrix PRIVATE resolve_logger resolve_vector)

# Link to CUDA ReSolve backend if CUDA is support enabled
if (RESOLVE_USE_CUDA)
  target_sources(resolve_matrix PRIVATE ${Matrix_CUDASDK_SRC})
  target_link_libraries(resolve_matrix PUBLIC resolve_backend_cuda)
endif()

if (RESOLVE_USE_HIP)
  target_sources(resolve_matrix PRIVATE ${Matrix_ROCM_SRC})
  target_link_libraries(resolve_matrix PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_matrix PUBLIC resolve_backend_cpu)
endif()

target_include_directories(resolve_matrix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${Matrix_HEADER_INSTALL} DESTINATION include/resolve/matrix)
