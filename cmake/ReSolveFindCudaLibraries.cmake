# Exports target `resolve_cuda` which finds all cuda libraries needed by
# resolve.

add_library(resolve_cuda INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(
  resolve_cuda INTERFACE CUDA::cusolver CUDA::cublas CUDA::cusparse
                         CUDA::cudart
)

if(RESOLVE_USE_CUDSS)
  target_link_libraries(resolve_cuda INTERFACE cudss)
endif()

if(RESOLVE_USE_PROFILING)
  if(TARGET CUDA::nvtx3)
    target_link_libraries(resolve_cuda INTERFACE CUDA::nvtx3)
    target_compile_definitions(resolve_cuda INTERFACE RESOLVE_USE_NVTX3)
  else()
    target_link_libraries(resolve_cuda INTERFACE CUDA::nvToolsExt)
  endif()
endif()

install(TARGETS resolve_cuda EXPORT ReSolveTargets)
