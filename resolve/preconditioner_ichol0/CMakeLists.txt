#[[

@brief Zero-fill incomplete Cholesky preconditioner implementation

@author Andrew Xu <xua1@ornl.gov>

]]

set(PreconditionerIChol0_SRC PreconditionerIChol0.cpp)

set(PreconditionerIChol0_HEADER_INSTALL PreconditionerIChol0.hpp PreconditionerIChol0Impl.hpp)

if(RESOLVE_USE_CUDSS)
  list(APPEND PreconditionerIChol0_SRC PreconditionerIChol0Cuda.cpp)
  list(APPEND PreconditionerIChol0_HEADER_INSTALL PreconditionerIChol0Cuda.hpp)
elseif(RESOLVE_USE_HIP)
  list(APPEND PreconditionerIChol0_SRC PreconditionerIChol0Hip.cpp)
  list(APPEND PreconditionerIChol0_HEADER_INSTALL PreconditionerIChol0Hip.hpp)
endif()

# Build shared library ReSolve
add_library(resolve_preconditioner_ichol0 OBJECT ${PreconditionerIChol0_SRC})

target_link_libraries(resolve_preconditioner_ichol0 PUBLIC resolve_tpl)

target_include_directories(
  resolve_preconditioner_ichol0
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<INSTALL_INTERFACE:include>
)

install(FILES ${PreconditionerIChol0_HEADER_INSTALL}
        DESTINATION include/resolve/preconditioner_ichol0
)
