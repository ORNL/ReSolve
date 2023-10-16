#pragma once

#include <string>

#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

namespace ReSolve
{
  LinAlgWorkspace* createLinAlgWorkspace(std::string memspace);
}
