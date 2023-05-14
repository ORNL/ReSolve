#pragma once
#include <string>
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"

namespace ReSolve {
  class resolveLinSolver {
    public:
      resolveLinSolver();
      ~resolveLinSolver();

      virtual resolveReal* solve(resolveMatrix* A, resolveReal* rhs);
      virtual setup(resolveMatrix* A);
  }
