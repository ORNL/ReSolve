set(Matrix_SRC io.cpp)


set(Matrix_SRC_CUDA
    #MatrixCoo.cpp
)

set(Matrix_HEADER_INSTALL
    #MatrixCoo.hpp
    io.hpp
)

set_source_files_properties(${Matrix_SRC_CUDA} PROPERTIES LANGUAGE CUDA)

# Build shared library ReSolve
add_library(resolve_matrix OBJECT io.cpp)
target_link_libraries(resolve_matrix PUBLIC resolve_cuda)

#install(TARGETS resolve_matrix EXPORT ReSolveTargets)

#add_library(ReSolve SHARED ${ReSolve_SRC} ${ReSolve_SRC_CUDA})

target_include_directories(resolve_matrix PUBLIC
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
install(FILES ${ReSolve_HEADER_INSTALL} DESTINATION include/resolve)

