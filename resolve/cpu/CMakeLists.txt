#[[

@brief Build Re::Solve backend when there is no GPU support

@author Slaven Peles <peless@ornl.gov>

]]

set(ReSolve_CPU_SRC MemoryUtils.cpp)

set(ReSolve_CPU_HEADER_INSTALL CpuMemory.hpp)

# First create dummy backend
add_library(resolve_backend_cpu SHARED ${ReSolve_CPU_SRC})
target_link_libraries(resolve_backend_cpu PRIVATE resolve_logger)

# install include headers
install(FILES ${ReSolve_CPU_HEADER_INSTALL} DESTINATION include/resolve/cpu)
