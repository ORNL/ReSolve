
set(Matrix_SRC 
    io.cpp
    Sparse.cpp
    Csr.cpp
    Csc.cpp
    Coo.cpp
    MatrixHandler.cpp
)


set(Matrix_HEADER_INSTALL
    io.hpp
    Sparse.hpp
    Coo.hpp
    Csr.hpp
    Csc.hpp
    MatrixHandler.hpp
)

# Build shared library ReSolve::matrix
add_library(resolve_matrix SHARED ${Matrix_SRC})
target_link_libraries(resolve_matrix PUBLIC resolve_backend_cuda)

#install(TARGETS resolve_matrix EXPORT ReSolveTargets)

target_include_directories(resolve_matrix INTERFACE
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
install(FILES ${Matrix_HEADER_INSTALL} DESTINATION include/resolve/matrix)

