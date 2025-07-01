#[[

@brief Build ReSolve Stopwatch class

@author Adham Ibrahim <ibrahimas@ornl.gov>

]]

set(STOPWATCH_SRC 
  Stopwatch.cpp
)

set(STOPWATCH_HEADER_INSTALL
  Stopwatch.hpp
)

# Build shared library ReSolve
add_library(resolve_stopwatch OBJECT ${STOPWATCH_SRC})
set_property(TARGET resolve_stopwatch PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(resolve_stopwatch PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${STOPWATCH_HEADER_INSTALL} DESTINATION include/resolve/utilities/stopwatch)
