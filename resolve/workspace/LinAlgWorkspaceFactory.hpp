#pragma once

#include <string>

#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

// #ifndef RESOLVE_USE_CUDA
// using ReSolve::LinAlgWorkspace = ReSolve::LinAlgWorkspaceCUDA;
// #endif

namespace ReSolve
{
  LinAlgWorkspace* createLinAlgWorkspace(std::string memspace);
}
