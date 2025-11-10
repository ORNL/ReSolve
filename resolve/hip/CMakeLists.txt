#[[

@brief Build ReSolve HIP backend

@author Slaven Peles <peless@ornl.gov>

]]

set(ReSolve_HIP_SRC hipKernels.hip hipSketchingKernels.hip hipVectorKernels.hip
                    MemoryUtils.hip
)

set(ReSolve_HIP_HEADER_INSTALL
    hipKernels.h hipSketchingKernels.h hipVectorKernels.h HipMemory.hpp
    hip_check_errors.hpp
)

set_source_files_properties(${ReSolve_HIP_SRC} PROPERTIES LANGUAGE HIP)

# First create HIP backend (this should really be HIP _API_ backend, separate
# backend will be needed for HIP SDK)
add_library(resolve_backend_hip SHARED ${ReSolve_HIP_SRC})
target_link_libraries(resolve_backend_hip PRIVATE resolve_logger)
target_link_libraries(resolve_backend_hip PUBLIC resolve_hip)

# install include headers
install(FILES ${ReSolve_HIP_HEADER_INSTALL} DESTINATION include/resolve/hip)
