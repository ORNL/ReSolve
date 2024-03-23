#include <resolve/matrix/Sparse.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include "LinSolver.hpp"


namespace ReSolve 
{
  using out = io::Logger;

  LinSolver::LinSolver()
  {
  }

  LinSolver::~LinSolver()
  {
    //destroy the matrix and hadlers
  }

  real_type LinSolver::evaluateResidual()
  {
    //to be implemented
    return 1.0;
  }

  //
  // Direct solver methods implementations
  //

  LinSolverDirect::LinSolverDirect()
  {
    L_ = nullptr;
    U_ = nullptr;
    P_ = nullptr;
    Q_ = nullptr;
  }

  LinSolverDirect::~LinSolverDirect()
  {
  }

  int LinSolverDirect::setup(matrix::Sparse* A,
                             matrix::Sparse* /* L */,
                             matrix::Sparse* /* U */,
                             index_type*     /* P */,
                             index_type*     /* Q */,
                             vector_type*  /* rhs */)
  {
    if (A == nullptr) {
      return 1;
    }
    this->A_ = A;
    return 0;
  }

  int LinSolverDirect::analyze()
  {
    return 1;
  } //the same as symbolic factorization

  int LinSolverDirect::factorize()
  {
    return 1;
  }

  int LinSolverDirect::refactorize()
  {
    return 1;
  }

  matrix::Sparse* LinSolverDirect::getLFactor()
  {
    return nullptr;
  } 
  
  matrix::Sparse* LinSolverDirect::getUFactor()
  {
    return nullptr;
  } 
  
  index_type*  LinSolverDirect::getPOrdering()
  {
    return nullptr;
  } 
  
  index_type*  LinSolverDirect::getQOrdering()
  {
    return nullptr;
  }

  void LinSolverDirect::setPivotThreshold(real_type tol)
  {
    pivot_threshold_tol_ = tol;
  }

  void LinSolverDirect::setOrdering(int ordering)
  {
    ordering_ = ordering;
  }

  void LinSolverDirect::setHaltIfSingular(bool is_halt)
  {
    halt_if_singular_ = is_halt;
  }

  real_type LinSolverDirect::getMatrixConditionNumber()
  {
    out::error() << "Solver does not implement returning system matrix condition number.\n";
    return -1.0;
  }

  //
  // Iterative solver methods implementations
  //

  LinSolverIterative::LinSolverIterative()
  {
  }
  
  LinSolverIterative::~LinSolverIterative()
  {
  }

  int LinSolverIterative::setup(matrix::Sparse* A)
  {
    if (A == nullptr) {
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


  real_type  LinSolverIterative::getTol()
  {
    return tol_;
  }

  index_type  LinSolverIterative::getMaxit()
  {
    return maxit_;
  }

  index_type  LinSolverIterative::getRestart()
  {
    return restart_;
  }

  index_type  LinSolverIterative::getConvCond()
  {
    return conv_cond_;
  }

  bool  LinSolverIterative::getFlexible()
  {
    return flexible_;
  }

  int LinSolverIterative::setOrthogonalization(GramSchmidt* /* gs */)
  {
    out::error() << "Solver does not implement setting orthogonalization.\n";
    return 1;
  }

  void  LinSolverIterative::setTol(real_type new_tol)
  {
    this->tol_ = new_tol;
  }

  void  LinSolverIterative::setMaxit(index_type new_maxit)
  {
    this->maxit_ = new_maxit;
  }

  // void  LinSolverIterative::setRestart(index_type new_restart)
  // {
  //   this->restart_ = new_restart;
  // }

  void  LinSolverIterative::setConvCond(index_type new_conv_cond)
  {
    this->conv_cond_ = new_conv_cond;
  }

  void  LinSolverIterative::setFlexible(bool new_flex)
  {
    this->flexible_ = new_flex;
  }
}



