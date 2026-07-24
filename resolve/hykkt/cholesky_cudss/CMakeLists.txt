#[[

@brief Build ReSolve matrix module

@author Slaven Peles <peless@ornl.gov>

]]

set(HyKKT_CHOL_CUDSS_SRC CholeskySolverCuDss.cpp CholeskySolverCuDssCuda.cpp)

# Header files to be installed
set(HyKKT_CHOL_CUDSS_HEADER_INSTALL CholeskySolverCuDss.hpp
                                    CholeskySolverCuDssCuda.hpp
)

add_library(resolve_hykkt_chol_cudss SHARED ${HyKKT_CHOL_CUDSS_SRC})

target_link_libraries(resolve_hykkt_chol_cudss PUBLIC ${suitesparse_cholmod})
target_include_directories(
  resolve_hykkt_chol_cudss PUBLIC ${SUITESPARSE_INCLUDE_DIR}
)

# Link to CUDA ReSolve backend if CUDA is support enabled
target_sources(resolve_hykkt_chol_cudss PRIVATE ${HyKKT_CHOL_CUDSS_SRC})
target_link_libraries(resolve_hykkt_chol_cudss PUBLIC resolve_backend_cuda)

target_link_libraries(
  resolve_hykkt_chol_cudss PUBLIC resolve_workspace resolve_vector
                                  resolve_matrix resolve_logger
)

target_include_directories(
  resolve_hykkt_chol_cudss INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                                     $<INSTALL_INTERFACE:include>
)

install(FILES ${HyKKT_CHOL_CUDSS_HEADER_INSTALL}
        DESTINATION include/resolve/hykkt
)
