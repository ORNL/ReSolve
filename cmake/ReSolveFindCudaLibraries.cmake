# Exports target `resolve_cuda` which finds all cuda libraries needed by resolve.


add_library(resolve_cuda INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(resolve_cuda INTERFACE
  CUDA::cusolver 
  CUDA::cublas
  CUDA::cusparse
  CUDA::cudart
  )

if(RESOLVE_USE_PROFILING)
  target_link_libraries(resolve_cuda INTERFACE
    CUDA::nvToolsExt
  )
endif()

install(TARGETS resolve_cuda EXPORT ReSolveTargets)
