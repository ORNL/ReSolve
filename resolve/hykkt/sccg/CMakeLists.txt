# Add source files for the SCCG module
set(SCCG_SRC SchurComplementConjugateGradient.cpp)
set(SCCG_HEADER_INSTALL SchurComplementConjugateGradient.hpp)

# Build shared library ReSolve::hykkt
add_library(resolve_hykkt_sccg SHARED ${SCCG_SRC})

target_link_libraries(
  resolve_hykkt_sccg
  PRIVATE resolve_hykkt resolve_hykkt_chol resolve_logger resolve_vector
          resolve_matrix ${suitesparse_cholmod}
)

# Link to CUDA ReSolve backend if CUDA is support enabled

target_include_directories(resolve_hykkt_sccg PUBLIC ${SUITESPARSE_INCLUDE_DIR})

if(RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt_sccg PRIVATE ${Matrix_CUDASDK_SRC})
  target_link_libraries(resolve_hykkt_sccg PUBLIC resolve_backend_cuda)
endif()

# Link to HIP ReSolve backend if HIP is support enabled.

if(RESOLVE_USE_HIP)
  target_sources(resolve_hykkt_sccg PRIVATE ${Matrix_ROCM_SRC})
  target_link_libraries(resolve_hykkt_sccg PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if(NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_hykkt_sccg PUBLIC resolve_backend_cpu)
endif()

target_include_directories(
  resolve_hykkt_sccg INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                               $<INSTALL_INTERFACE:include>
)

install(FILES ${SCCG_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
