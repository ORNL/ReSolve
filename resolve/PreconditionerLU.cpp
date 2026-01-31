/**
 * @file   PreconditionerLU.cpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Declaration of preconditioner ILU0 class.
 *
 */

#include "PreconditionerLU.hpp"

#include <resolve/LinSolverDirect.hpp>

namespace ReSolve
{
  /**
   * @brief Constructor for PreconditionerLU.
   *
   * @param[in] solver - Pointer to the LinSolverDirect object.
   */
  PreconditionerLU::PreconditionerLU(LinSolverDirect* solver)
  {
    solver_ = solver;
  }

  /**
   * @brief Destructor for PreconditionerLU
   */
  PreconditionerLU::~PreconditionerLU()
  {
  }

  /**
   * @brief Sets up the preconditioner with the given matrix
   *
   * @param[in] A - System matrix to set up the preconditioner with
   *
   * @return int 0 if successful, 1 if it fails
   */
  int PreconditionerLU::setup(matrix_type* A)
  {
    if (A == nullptr)
    {
      return 1;
    }
    solver_->setup(A);

    return 0;
  }

  /**
   * @brief Applies the preconditioner to solve the system Mx = rhs
   *
   * Computes x = M^(-1) * rhs where M is the preconditioner matrix.
   *
   * @param[in] rhs - Right-hand-side vector
   * @param[in] x   - Solution vector
   *
   * @return int 0 if successful, 1 if fails
   */
  int PreconditionerLU::apply(vector_type* rhs, vector_type* x)
  {
    if (solver_ == nullptr)
    {
      return 1;
    }
    solver_->solve(rhs, x);

    return 0;
  }

  /**
   * @brief Resets the preconditioner with the given matrix
   *
   * @param[in] A - System matrix to reset the preconditioner with
   *
   * @return int 0 if successful, 1 if it fails
   */
  int PreconditionerLU::reset(matrix_type* A)
  {
    if (A == nullptr)
    {
      return 1;
    }
    solver_->reset(A);

    return 0;
  }
} // namespace ReSolve
