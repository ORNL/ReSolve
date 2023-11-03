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
    VectorKernels.hpp
)

# Add CUDA vector handler if CUDA support is enabled
if(RESOLVE_USE_CUDA)
  set(Vector_SRC ${Vector_SRC} ${Vector_CUDASDK_SRC})
endif()

# and hip
if(RESOLVE_USE_HIP)
  set(Vector_SRC ${Vector_SRC} ${Vector_ROCM_SRC})
endif()

add_library(resolve_vector SHARED ${Vector_SRC})
target_link_libraries(resolve_vector PRIVATE resolve_logger)

# Link to ReSolve CUDA backend if CUDA is enabled
if (RESOLVE_USE_CUDA)
  target_link_libraries(resolve_vector PUBLIC resolve_backend_cuda)
endif()

if (RESOLVE_USE_HIP)
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

# # TODO: Make this PRIVATE dependency (requires refactoring ReSolve code)
# target_link_libraries(ReSolve PUBLIC resolve_tpl)

# install(TARGETS ReSolve
#         EXPORT ReSolveTargets
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib)
# install include headers
install(FILES ${Vector_HEADER_INSTALL} DESTINATION include/resolve/vector)

