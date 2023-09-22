#[[

@brief Build ReSolve output log system

@author Slaven Peles <peless@ornl.gov>

]]

set(Logger_SRC 
  Logger.cpp
)

set(Logger_HEADER_INSTALL
  Logger.hpp
)

# Build shared library ReSolve
add_library(resolve_logger SHARED ${Logger_SRC})

target_include_directories(resolve_logger INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${Matrix_HEADER_INSTALL} DESTINATION include/resolve/utilities/logger)
