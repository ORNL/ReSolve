#[[

@brief Build ReSolve function that returns version at runtime

@author Slaven Peles <peless@ornl.gov>

]]

set(Logger_SRC 
  version.cpp
)

set(Logger_HEADER_INSTALL
  version.hpp
)

# Build shared library ReSolve
add_library(resolve_version OBJECT ${Logger_SRC})
set_property(TARGET resolve_version PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(resolve_version PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${Logger_HEADER_INSTALL} DESTINATION include/resolve/utilities/version)
