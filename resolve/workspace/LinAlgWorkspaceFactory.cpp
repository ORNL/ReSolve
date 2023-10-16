#pragma once

#include "LinAlgWorkspaceFactory.hpp"

namespace ReSolve
{
  /// @brief  Workspace factory
  /// @param[in] memspace memory space ID 
  /// @return pointer to the linear algebra workspace
  LinAlgWorkspace* createLinAlgWorkspace(std::string memspace)
  {
    if (memspace == "cuda") {
#ifdef RESOLVE_USE_CUDA
      LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
      workspace->initializeHandles();
      return workspace;
#else
      return nullptr;
#endif
    } 
    // If not CUDA, return default
    return (new LinAlgWorkspace());
  }

}
