#[[

@brief Build ReSolve CUDA backend

@author Slaven Peles <peless@ornl.gov>

]]

set(ReSolve_CUDA_SRC
    cudaKernels.cu
    cudaSketchingKernels.cu
    cudaVectorKernels.cu
    MemoryUtils.cu
)

set(ReSolve_CUDA_HEADER_INSTALL
    cudaKernels.h
    cudaSketchingKernels.h
    cudaVectorKernels.h
    CudaMemory.hpp
    cuda_check_errors.hpp
)

set_source_files_properties(${ReSolve_CUDA_SRC} PROPERTIES LANGUAGE CUDA)

# First create CUDA backend 
# (this should really be CUDA _API_ backend, 
# separate backend will be needed for CUDA SDK)
add_library(resolve_backend_cuda SHARED ${ReSolve_CUDA_SRC})
target_link_libraries(resolve_backend_cuda PRIVATE resolve_logger)
target_link_libraries(resolve_backend_cuda PUBLIC resolve_cuda)

# install include headers
install(FILES ${ReSolve_CUDA_HEADER_INSTALL} DESTINATION include/resolve/cuda)
