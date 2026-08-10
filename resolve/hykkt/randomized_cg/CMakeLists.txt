# Add source files for the RandomizedCG module
set(RandomizedCG_SRC RandomizedConjugateGradient.cpp RandomizedConjugateGradientCuda.cu RandomizedConjugateGradientHip.hip)
set(RandomizedCG_HEADER_INSTALL RandomizedConjugateGradient.hpp RandomizedConjugateGradientImpl.hpp RandomizedConjugateGradientCuda.hpp RandomizedConjugateGradientHip.hpp)

# Build shared library ReSolve::hykkt
add_library(resolve_hykkt_randomized_cg SHARED ${RandomizedCG_SRC})

target_link_libraries(
  resolve_hykkt_randomized_cg
  PRIVATE ReSolve resolve_hykkt resolve_hykkt_chol resolve_logger resolve_vector
          resolve_matrix ${suitesparse_cholmod}
)

# Link to CUDA ReSolve backend if CUDA is support enabled

target_include_directories(resolve_hykkt_randomized_cg PUBLIC ${SUITESPARSE_INCLUDE_DIR})

if(RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt_randomized_cg PRIVATE ${Matrix_CUDASDK_SRC})
  target_link_libraries(resolve_hykkt_randomized_cg PUBLIC resolve_backend_cuda)
endif()

# Link to HIP ReSolve backend if HIP is support enabled.

if(RESOLVE_USE_HIP)
  target_sources(resolve_hykkt_randomized_cg PRIVATE ${Matrix_ROCM_SRC})
  target_link_libraries(resolve_hykkt_randomized_cg PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if(NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_hykkt_randomized_cg PUBLIC resolve_backend_cpu)
endif()

target_include_directories(
  resolve_hykkt_randomized_cg INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                               $<INSTALL_INTERFACE:include>
)

target_compile_definitions(resolve_hykkt_randomized_cg 
    PRIVATE HIP_ENABLE_WARP_SYNC_BUILTINS=1
)

install(FILES ${RandomizedCG_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
