#[[

@brief Build ReSolve workspace module

@author Slaven Peles <peless@ornl.gov>

]]

# C++ code
set(ReSolve_Workspace_SRC
  LinAlgWorkspaceCpu.cpp
)

# C++ code that depends on CUDA SDK libraries
set(ReSolve_Workspace_CUDASDK_SRC
    LinAlgWorkspaceCUDA.cpp
)

set(ReSolve_Workspace_ROCM_SRC
  LinAlgWorkspaceHIP.cpp
)

set(ReSolve_Workspace_HEADER_INSTALL
  LinAlgWorkspace.hpp
  LinAlgWorkspaceCpu.hpp
  LinAlgWorkspaceCUDA.hpp
  LinAlgWorkspaceHIP.hpp
)

add_library(resolve_workspace SHARED ${ReSolve_Workspace_SRC})

# If cuda is enabled, add CUDA SDK workspace files
if(RESOLVE_USE_CUDA)
  target_sources(resolve_workspace PRIVATE ${ReSolve_Workspace_CUDASDK_SRC})
  target_link_libraries(resolve_workspace PUBLIC resolve_backend_cuda)
endif()

if(RESOLVE_USE_HIP)
  target_sources(resolve_workspace PRIVATE ${ReSolve_Workspace_ROCM_SRC})
  target_link_libraries(resolve_workspace PUBLIC resolve_backend_hip)
endif()

target_include_directories(resolve_workspace PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)

# install include headers
install(FILES ${ReSolve_Workspace_HEADER_INSTALL} DESTINATION include/resolve/workspace)
