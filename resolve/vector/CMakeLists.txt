
set(Vector_SRC 
    VectorHandler.cpp
)



set(Vector_SRC_CUDA
    Vector.cpp
)

set(Vector_HEADER_INSTALL
    Vector.hpp
    VectorHandler.hpp
)

set_source_files_properties(${Vector_SRC_CUDA} PROPERTIES LANGUAGE CUDA)

# Build shared library ReSolve
add_library(resolve_vector SHARED ${Vector_SRC} ${Vector_SRC_CUDA})
target_link_libraries(resolve_vector PUBLIC resolve_cuda)

#install(TARGETS resolve_vector EXPORT ReSolveTargets)

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

