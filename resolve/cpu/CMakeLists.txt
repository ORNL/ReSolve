#[[

@brief Build ReSolve backend when there is no GPU support

@author Slaven Peles <peless@ornl.gov>

]]

set(ReSolve_CPU_SRC
    MemoryUtils.cpp
    cpuVectorKernels.cpp
)

set(ReSolve_CPU_HEADER_INSTALL
    CpuMemory.hpp
)

# First create dummy backend
add_library(resolve_backend_cpu SHARED ${ReSolve_CPU_SRC})
target_include_directories(resolve_backend_cpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# install include headers
install(FILES ${ReSolve_CPU_HEADER_INSTALL} DESTINATION include/resolve/cpu)
