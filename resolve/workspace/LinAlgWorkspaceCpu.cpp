#include "LinAlgWorkspaceCpu.hpp"

#include "resolve/Common.hpp"

#include <cstddef>

namespace ReSolve
{
  LinAlgWorkspaceCpu::LinAlgWorkspaceCpu()
    : generator_(constants::SEED)
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

  std::mt19937& LinAlgWorkspaceCpu::getRng()
  {
    return generator_;
  }
} // namespace ReSolve
