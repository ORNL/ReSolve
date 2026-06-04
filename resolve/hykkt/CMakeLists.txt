#[[

@brief Build HyKKT solver module

@author Andrew Xu <xua1@ornl.gov>

]]

add_subdirectory(permutation)
add_subdirectory(ruiz)
add_subdirectory(cholesky)
add_subdirectory(spgemm)
add_subdirectory(sccg)

# Build shared library ReSolve::hykkt
set(HYKKT_SOLVER_SRC HyKKTSolver.cpp)
set(HYKKT_SOLVER_HEADER_INSTALL HyKKTSolver.hpp)

# Build shared library ReSolve::hykkt
add_library(resolve_hykkt_solver SHARED ${HYKKT_SOLVER_SRC})

target_link_libraries(
  resolve_hykkt_solver
  PRIVATE resolve_hykkt
          resolve_hykkt_ruiz
          resolve_hykkt_chol
          resolve_hykkt_spgemm
          resolve_hykkt_sccg
          resolve_logger
          resolve_vector
          resolve_matrix
)

# Link to CUDA ReSolve backend if CUDA is support enabled.

# target_include_directories(resolve_hykkt_solver PUBLIC
# ${SUITESPARSE_INCLUDE_DIR})

if(RESOLVE_USE_CUDA)
  target_sources(resolve_hykkt_solver PRIVATE ${Matrix_CUDASDK_SRC})
  target_link_libraries(resolve_hykkt_solver PUBLIC resolve_backend_cuda)
endif()

# Link to HIP ReSolve backend if HIP is support enabled

if(RESOLVE_USE_HIP)
  target_sources(resolve_hykkt_solver PRIVATE ${Matrix_ROCM_SRC})
  target_link_libraries(resolve_hykkt_solver PUBLIC resolve_backend_hip)
endif()

# Link to dummy device backend if GPU support is not enabled
if(NOT RESOLVE_USE_GPU)
  target_link_libraries(resolve_hykkt_solver PUBLIC resolve_backend_cpu)
endif()

target_include_directories(
  resolve_hykkt_solver INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                                 $<INSTALL_INTERFACE:include>
)

install(FILES ${HYKKT_SOLVER_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
