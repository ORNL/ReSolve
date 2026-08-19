#[[

@brief Zero-fill incomplete Cholesky preconditioner implementation

@author Andrew Xu <xua1@ornl.gov>

]]

set(PreconditionerIChol0_SRC PreconditionerIChol0.cpp PreconditionerIChol0Hip.cpp)

set(PreconditionerIChol0_HEADER_INSTALL PreconditionerIChol0.hpp PreconditionerIChol0Impl.hpp PreconditionerIChol0Hip.hpp)

# Build shared library ReSolve
add_library(resolve_preconditioner_ichol0 OBJECT ${PreconditionerIChol0_SRC})

target_include_directories(
  resolve_preconditioner_ichol0
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<INSTALL_INTERFACE:include>
)

install(FILES ${PreconditionerIChol0_HEADER_INSTALL}
        DESTINATION include/resolve/preconditioner_ichol0
)
