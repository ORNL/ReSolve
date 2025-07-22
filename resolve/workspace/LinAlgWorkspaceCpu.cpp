#include "LinAlgWorkspaceCpu.hpp"

#include <cstddef>

namespace ReSolve
{
  LinAlgWorkspaceCpu::LinAlgWorkspaceCpu()
  {
  }

  LinAlgWorkspaceCpu::~LinAlgWorkspaceCpu()
  {
  }

  void LinAlgWorkspaceCpu::initializeHandles()
  {
  }

  void LinAlgWorkspaceCpu::resetLinAlgWorkspace()
  {
    // No resources to reset in CPU workspace
    return;
  }
} // namespace ReSolve
