#[[

@brief Build ReSolve workspace module

@author Slaven Peles <peless@ornl.gov>

]]

set(ReSolve_Workspace_SRC
  LinAlgWorkspace.cpp
  LinAlgWorkspaceFactory.cpp
)

set(ReSolve_Workspace_CUDA_SRC # consider adding a separate dir for cuda-sdk dependent code
    LinAlgWorkspaceCUDA.cpp
)

set(ReSolve_Workspace_HEADER_INSTALL
  LinAlgWorkspace.hpp
  LinAlgWorkspaceFactory.hpp
  LinAlgWorkspaceCUDA.hpp
)

if(RESOLVE_USE_CUDA)
  add_library(resolve_workspace SHARED ${ReSolve_Workspace_SRC} ${ReSolve_Workspace_CUDA_SRC})
  target_link_libraries(resolve_workspace PUBLIC resolve_backend_cuda)
else(RESOLVE_USE_CUDA)  
  add_library(resolve_workspace SHARED ${ReSolve_Workspace_SRC})
endif(RESOLVE_USE_CUDA)

target_include_directories(resolve_workspace INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# install include headers
install(FILES ${ReSolve_Workspace_HEADER_INSTALL} DESTINATION include/resolve/workspace)
