#[[

@brief Build ReSolve matrix module

@author Slaven Peles <peless@ornl.gov>

]]

# C++ code
set(HyKKT_SRC
    Permutation.cpp
    PermutationHandler.cpp
    cpuPermutationKernels.cpp
    CudaPermutationKernels.cu
    HipPermutationKernels.hip
)

# C++ code that depends on CUDA SDK libraries
set(HyKKT_CUDASDK_SRC
)

# and on HIP
set(HyKKT_ROCM_SRC
)

# Header files to be installed
set(HyKKT_HEADER_INSTALL
    Permutation.hpp
    cpuPermutationKernels.hpp
    CudaPermutationKernels.hpp
)

# Build shared library ReSolve::matrix
add_library(resolve_hykkt SHARED ${HyKKT_SRC})
target_link_libraries(resolve_hykkt PUBLIC resolve_logger resolve_vector ${suitesparse_amd})
target_include_directories(resolve_hykkt PUBLIC ${SUITESPARSE_INCLUDE_DIR})

# Link to CUDA ReSolve backend if CUDA is support enabled
if (RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt PRIVATE ${HyKKT_CUDASDK_SRC})
  target_link_libraries(resolve_hykkt PUBLIC resolve_backend_cuda)
endif()

if (RESOLVE_USE_HIP)
  target_sources(resolve_hykkt PRIVATE ${HyKKT_ROCM_SRC})
  target_link_libraries(resolve_hykkt PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_hykkt PUBLIC resolve_backend_cpu)
endif()

target_link_libraries(resolve_hykkt PUBLIC resolve_workspace)

target_include_directories(resolve_hykkt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${HyKKT_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
