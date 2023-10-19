#[[

@brief Build ReSolve (multi)vector module

@author Slaven Peles <peless@ornl.gov>

]]


set(Vector_SRC 
    Vector.cpp
    VectorHandler.cpp
    VectorHandlerCpu.cpp
    VectorHandlerCuda.cpp
)


set(Vector_HEADER_INSTALL
    Vector.hpp
    VectorHandler.hpp
    VectorKernels.hpp
)

add_library(resolve_vector SHARED ${Vector_SRC})

if (RESOLVE_USE_CUDA)
    target_link_libraries(resolve_vector PUBLIC resolve_backend_cuda)
else()
    target_link_libraries(resolve_vector PUBLIC resolve_backend_cpu)
endif()

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

