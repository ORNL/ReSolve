#[[

@brief Build ruiz module for HyKKT solver

@author Adham Ibrahim <ibrahimas@ornl.gov>

]]

# Add source files for the Ruiz module
set(RUIZ_SRC
  RuizScaling.cpp
  RuizScalingKernelsCPU.cpp
  RuizScalingKernelsHIP.hip
  RuizScalingKernelsCUDA.cu
)
set(RUIZ_HEADER_INSTALL
  RuizScaling.hpp
  RuizScalingKernels.hpp
  RuizScalingKernelsCPU.hpp
  RuizScalingKernelsHIP.hpp
  RuizScalingKernelsCUDA.hpp
)

# Build shared library ReSolve::hykkt
add_library(resolve_hykkt_ruiz SHARED ${RUIZ_SRC})

target_link_libraries(resolve_hykkt_ruiz PRIVATE resolve_logger resolve_vector)

# Link to CUDA ReSolve backend if CUDA is support enabled
if (RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt_ruiz PRIVATE ${Matrix_CUDASDK_SRC})
  target_link_libraries(resolve_hykkt_ruiz PUBLIC resolve_backend_cuda)
endif()

if (RESOLVE_USE_HIP)
  target_sources(resolve_hykkt_ruiz PRIVATE ${Matrix_ROCM_SRC})
  target_link_libraries(resolve_hykkt_ruiz PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_hykkt_ruiz PUBLIC resolve_backend_cpu)
endif()

target_include_directories(resolve_hykkt_ruiz INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${RUIZ_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
