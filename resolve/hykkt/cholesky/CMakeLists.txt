#[[

@brief Build ReSolve matrix module

@author Slaven Peles <peless@ornl.gov>

]]

# C++ code
set(HyKKT_CHOL_SRC
    CholeskySolver.cpp
    CholeskySolverCpu.cpp
)

# C++ code that depends on CUDA SDK libraries
set(HyKKT_CUDASDK_SRC
    CholeskySolverCuda.cpp
)

# and on HIP
set(HyKKT_ROCM_SRC
    CholeskySolverHip.cpp
)

# Header files to be installed
set(HyKKT_CHOL_HEADER_INSTALL
    CholeskySolver.hpp
    CholeskySolverImpl.hpp
    CholeskySolverCpu.hpp
    CholeskySolverCuda.hpp
    CholeskySolverHip.hpp
)

add_library(resolve_hykkt_chol SHARED ${HyKKT_CHOL_SRC})

target_link_libraries(resolve_hykkt_chol PUBLIC ${suitesparse_cholmod})
target_include_directories(resolve_hykkt_chol PUBLIC ${SUITESPARSE_INCLUDE_DIR})

# Link to CUDA ReSolve backend if CUDA is support enabled
if (RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt_chol PRIVATE ${HyKKT_CUDASDK_SRC})
  target_link_libraries(resolve_hykkt_chol PUBLIC resolve_backend_cuda)
endif()

if (RESOLVE_USE_HIP)
  target_sources(resolve_hykkt_chol PRIVATE ${HyKKT_ROCM_SRC})
  target_link_libraries(resolve_hykkt_chol PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if (NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_hykkt_chol PUBLIC resolve_backend_cpu)
endif()

target_link_libraries(resolve_hykkt_chol PUBLIC resolve_workspace
                                                resolve_vector
                                                resolve_matrix
                                                resolve_logger)

target_include_directories(resolve_hykkt_chol INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${HyKKT_CHOL_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
