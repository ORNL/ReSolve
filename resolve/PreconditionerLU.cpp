/**
 * @file   PreconditionerLU.cpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Declaration of preconditioner ILU0 class.
 *
 */

#include <resolve/LinSolverDirect.hpp>
#include "PreconditionerLU.hpp"

namespace ReSolve
{
  PreconditionerLU::PreconditionerLU(LinSolverDirect* solver)
  {
    solver_ = solver;
  }

  PreconditionerLU::~PreconditionerLU()
  {
  }

  int PreconditionerLU::setup(matrix::Sparse* A)
  {
    if (A == nullptr)
    {
      return 1;
    }
    solver_->setup(A);

    return 0;
  }

  int PreconditionerLU::reset(matrix::Sparse* A)
  {
    if (solver_ == nullptr || A == nullptr)
    {
      return 1;
    }
    // LinSolverDirect doesn't have reset, so call setup instead
    return solver_->setup(A);
  }

  int PreconditionerLU::apply(vector_type* rhs, vector_type* x)
  {
    if (solver_ == nullptr)
    {
      return 1;
    }
    solver_->solve(rhs, x);

    return 0;
  }
} // namespace ReSolve