#[[

@brief Build ruiz module for HyKKT solver

@author Adham Ibrahim <ibrahimas@ornl.gov>

]]

# Add source files for the Ruiz module
set(RUIZ_SRC
  RuizScaler.cpp
  RuizScalingKernelsCPU.cpp
)
set(RUIZ_HEADER_INSTALL
  RuizScaler.hpp
  RuizScalingKernels.hpp
  RuizScalingKernelsCPU.hpp
)

# Build shared library ReSolve::hykkt
add_library(resolve_hykkt_ruiz SHARED ${RUIZ_SRC})

target_include_directories(resolve_hykkt_ruiz INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES ${RUIZ_HEADER_INSTALL} DESTINATION include/resolve/hykkt)
