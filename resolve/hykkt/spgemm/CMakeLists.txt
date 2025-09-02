#[[

@brief Build SpGEMM module for HyKKT solver

@author Adham Ibrahim <ibrahimas@ornl.gov>

]]

# Add source files for the SpGEMM module
set(SPGEMM_SRC
  SpGEMM.cpp
  SpGEMMCpu.cpp
)
set(SPGEMM_HEADER_INSTALL
  SpGEMM.hpp
  SpGEMMImpl.hpp
  SpGEMMCpu.hpp
  SpGEMMCuda.hpp
  SpGEMMHip.hpp
)

set(SPGEMM_CUDA_SRC
  SpGEMMCuda.cpp
)

set(SPGEMM_ROCM_SRC
  SpGEMMHip.cpp
)

# Build shared library ReSolve::hykkt
add_library(resolve_hykkt_spgemm SHARED ${SPGEMM_SRC})

target_link_libraries(resolve_hykkt_spgemm PRIVATE resolve_logger ${suitesparse_cholmod})
target_include_directories(resolve_hykkt_spgemm PUBLIC ${SUITESPARSE_INCLUDE_DIR})

if (RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt_spgemm PRIVATE ${SPGEMM_CUDA_SRC})
  target_link_libraries(resolve_hykkt_spgemm PUBLIC resolve_backend_cuda)
endif()

if (RESOLVE_USE_HIP)
  target_sources(resolve_hykkt_spgemm PRIVATE ${SPGEMM_ROCM_SRC})
  target_link_libraries(resolve_hykkt_spgemm PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_hykkt_spgemm PUBLIC resolve_backend_cpu)
endif()

target_include_directories(resolve_hykkt_spgemm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${SPGEMM_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
