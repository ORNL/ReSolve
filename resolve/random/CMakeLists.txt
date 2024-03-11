#[[

@brief Build ReSolve random sketching module

@author Slaven Peles <peless@ornl.gov>

]]

# C++ code
set(Random_SRC 
    RandomSketchingCountCpu.cpp
    RandomSketchingFWHTCpu.cpp
    cpuSketchingKernels.cpp
    SketchingHandler.cpp
)

# C++ code that depends on CUDA SDK libraries
set(Random_CUDASDK_SRC 
    RandomSketchingCountCuda.cpp
    RandomSketchingFWHTCuda.cpp
)

# and on HIP
set(Random_ROCM_SRC 
    RandomSketchingCountHip.cpp
    RandomSketchingFWHTHip.cpp
)

# Header files to be installed
set(Random_HEADER_INSTALL
    RandomSketchingImpl.hpp
    RandomSketchingCountCpu.hpp
    RandomSketchingFWHTCpu.hpp
    cpuSketchingKernels.h
    SketchingHandler.hpp
)

# Header files to be installed when HIP support is enabled
set(Random_ROCM_HEADER_INSTALL
  RandomSketchingCountHip.hpp
  RandomSketchingFWHTHip.hpp
)

# Header files to be installed when HIP support is enabled
set(Random_CUDASDK_HEADER_INSTALL
  RandomSketchingCountCuda.hpp
  RandomSketchingFWHTCuda.hpp
)


# Build shared library ReSolve::random
add_library(resolve_random SHARED ${Random_SRC})
target_link_libraries(resolve_random PRIVATE resolve_logger resolve_vector)

# Link to CUDA ReSolve backend if CUDA is support enabled
if (RESOLVE_USE_CUDA)
  target_sources(resolve_random PRIVATE ${Random_CUDASDK_SRC})
  target_link_libraries(resolve_random PUBLIC resolve_backend_cuda)
  list(APPEND Random_HEADER_INSTALL ${Random_CUDASDK_HEADER_INSTALL})
endif()

if (RESOLVE_USE_HIP)
  target_sources(resolve_random PRIVATE ${Random_ROCM_SRC})
  target_link_libraries(resolve_random PUBLIC resolve_backend_hip)
  list(APPEND Random_HEADER_INSTALL ${Random_ROCM_HEADER_INSTALL})
endif()

# Link to dummy device backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_random PUBLIC resolve_backend_cpu)
endif()

target_include_directories(resolve_random INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${Random_HEADER_INSTALL} DESTINATION include/resolve/random)
