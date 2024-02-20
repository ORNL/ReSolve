#[[

@brief Build ReSolve function that returns version at runtime

@author Slaven Peles <peless@ornl.gov>

]]

set(Options_SRC 
  CliOptions.cpp
)

set(Options_HEADER_INSTALL
  CliOptions.hpp
)

# Build shared library ReSolve
add_library(resolve_options OBJECT ${Options_SRC})
set_property(TARGET resolve_options PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(resolve_options PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${Options_HEADER_INSTALL} DESTINATION include/resolve/utilities/options)
