/**
 * @file LinSolverIterative.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Implementation of iterative solver base class.
 *
 */
#include <resolve/LinSolverDirect.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve
{
  using out = io::Logger;

  LinSolverIterative::LinSolverIterative()
  {
  }

  LinSolverIterative::~LinSolverIterative()
  {
  }

  int LinSolverIterative::setup(matrix::Sparse* A)
  {
    if (A == nullptr)
    {
      return 1;
    }
    this->A_ = A;
    return 0;
  }

  real_type LinSolverIterative::getFinalResidualNorm() const
  {
    return final_residual_norm_;
  }

  real_type LinSolverIterative::getInitResidualNorm() const
  {
    return initial_residual_norm_;
  }

  index_type LinSolverIterative::getNumIter() const
  {
    return total_iters_;
  }

  real_type LinSolverIterative::getTol() const
  {
    return tol_;
  }

  index_type LinSolverIterative::getMaxit() const
  {
    return maxit_;
  }

  int LinSolverIterative::setOrthogonalization(GramSchmidt* /* gs */)
  {
    out::error() << "Solver does not implement setting orthogonalization.\n";
    return 1;
  }

  void LinSolverIterative::setTol(real_type new_tol)
  {
    this->tol_ = new_tol;
  }

  void LinSolverIterative::setMaxit(index_type new_maxit)
  {
    this->maxit_ = new_maxit;
  }
} // namespace ReSolve
