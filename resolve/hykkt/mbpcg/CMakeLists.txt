# Add source files for the MBPCG module
set(MBPCG_SRC MultiBasisParallelConjugateGradient.cpp MultiBasisParallelConjugateGradientCuda.cu MultiBasisParallelConjugateGradientHip.hip SpMMCuda.cu SpMMHip.hip)
set(MBPCG_HEADER_INSTALL MultiBasisParallelConjugateGradient.hpp MultiBasisParallelConjugateGradientImpl.hpp MultiBasisParallelConjugateGradientCuda.hpp MultiBasisParallelConjugateGradientHip.hpp SpMMCuda.hip SpMMHip.hip)

# Build shared library ReSolve::hykkt
add_library(resolve_hykkt_mbpcg SHARED ${MBPCG_SRC})

target_link_libraries(
  resolve_hykkt_mbpcg
  PRIVATE ReSolve resolve_hykkt resolve_hykkt_chol resolve_logger resolve_vector
          resolve_matrix ${suitesparse_cholmod}
)

# Link to CUDA ReSolve backend if CUDA is support enabled

target_include_directories(resolve_hykkt_mbpcg PUBLIC ${SUITESPARSE_INCLUDE_DIR})

if(RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt_mbpcg PRIVATE ${Matrix_CUDASDK_SRC})
  target_link_libraries(resolve_hykkt_mbpcg PUBLIC resolve_backend_cuda)
endif()

# Link to HIP ReSolve backend if HIP is support enabled.

if(RESOLVE_USE_HIP)
  target_sources(resolve_hykkt_mbpcg PRIVATE ${Matrix_ROCM_SRC})
  target_link_libraries(resolve_hykkt_mbpcg PUBLIC resolve_backend_hip)
endif()

target_include_directories(
  resolve_hykkt_mbpcg INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                               $<INSTALL_INTERFACE:include>
)

target_compile_definitions(resolve_hykkt_mbpcg 
    PRIVATE HIP_ENABLE_WARP_SYNC_BUILTINS=1
)

install(FILES ${MBPCG_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
