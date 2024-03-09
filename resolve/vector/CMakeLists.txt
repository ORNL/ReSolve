#[[

@brief Build ReSolve (multi)vector module

@author Slaven Peles <peless@ornl.gov>

]]

# C++ code
set(Vector_SRC 
    Vector.cpp
    VectorHandler.cpp
    VectorHandlerCpu.cpp
)

# C++ code that depends on CUDA SDK libraries
set(Vector_CUDASDK_SRC 
    VectorHandlerCuda.cpp
)

#and hip
set(Vector_ROCM_SRC 
    VectorHandlerHip.cpp
)

# Header files to be installed
set(Vector_HEADER_INSTALL
    Vector.hpp
    VectorHandler.hpp
)

add_library(resolve_vector SHARED ${Vector_SRC})
target_link_libraries(resolve_vector PRIVATE resolve_logger)

# Add CUDA vector handler if CUDA support is enabled
if(RESOLVE_USE_CUDA)
  target_sources(resolve_vector PRIVATE ${Vector_CUDASDK_SRC})
  target_link_libraries(resolve_vector PUBLIC resolve_backend_cuda)
endif()

# and hip
if(RESOLVE_USE_HIP)
  target_sources(resolve_vector PRIVATE ${Vector_ROCM_SRC})
  target_link_libraries(resolve_vector PUBLIC resolve_backend_hip)
endif()

# If no GPU is enabled link to dummy device backend
if(NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_vector PUBLIC resolve_backend_cpu)
endif(NOT RESOLVE_USE_GPU)

target_include_directories(resolve_vector INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${Vector_HEADER_INSTALL} DESTINATION include/resolve/vector)
