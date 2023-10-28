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

# If cuda is enabled, add CUDA SDK workspace files
if(RESOLVE_USE_CUDA)
  set(ReSolve_Workspace_SRC ${ReSolve_Workspace_SRC} ${ReSolve_Workspace_CUDASDK_SRC})
endif()

if(RESOLVE_USE_HIP)
  set(ReSolve_Workspace_SRC ${ReSolve_Workspace_SRC} ${ReSolve_Workspace_ROCM_SRC})
endif()

add_library(resolve_workspace SHARED ${ReSolve_Workspace_SRC})

# If CUDA is enabled, link to ReSolve CUDA backend
if(RESOLVE_USE_CUDA)
  target_link_libraries(resolve_workspace PUBLIC resolve_backend_cuda)
endif(RESOLVE_USE_CUDA)  

if(RESOLVE_USE_HIP)
  target_link_libraries(resolve_workspace PUBLIC resolve_backend_hip)
endif(RESOLVE_USE_HIP)  

target_include_directories(resolve_workspace INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# install include headers
install(FILES ${ReSolve_Workspace_HEADER_INSTALL} DESTINATION include/resolve/workspace)
