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

  int LinAlgWorkspaceCpu::initializeHandles()
  {
    return 0;
  }

  void LinAlgWorkspaceCpu::resetLinAlgWorkspace()
  {
    // No resources to reset in CPU workspace
    return;
  }
} // namespace ReSolve
