#include <resolve/matrix/Sparse.hpp>
#include "LinSolver.hpp"


namespace ReSolve 
{
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
    factors_extracted_ = false;
  }

  LinSolverDirect::~LinSolverDirect()
  {
    delete L_;
    delete U_;
    delete [] P_;
    delete [] Q_;
  }

  int LinSolverDirect::setParameters()
  {
    return 1;
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
    factors_extracted_ = false;
    return 1;
  }

  int LinSolverDirect::refactorize()
  {
    factors_extracted_ = false;
    return 1;
  }

  int LinSolverDirect::solve(vector_type* /* rhs */, vector_type* /* x */) 
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
    pivotThreshold_ = tol;
  }

  void LinSolverDirect::setOrdering(int ordering)
  {
    ordering_ = ordering;
  }

  void LinSolverDirect::setHaltIfSingular(bool isHalt)
  {
    haltIfSingular_ = isHalt;
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

  int LinSolverIterative::resetMatrix(matrix::Sparse* /* A */)
  {
    A_ = nullptr;
    return -1;
  }

  int LinSolverIterative::setupPreconditioner(std::string /* type */, LinSolverDirect* /* solver */)
  {
    return -1;
  }

  int LinSolverIterative::solve(vector_type* /* rhs */, vector_type* /* init_guess */)
  {
    return 0;
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

  bool  LinSolverIterativeFGMRES::getFlexible()
  {
    return flexible_;
  }

  void  LinSolverIterative::setTol(real_type new_tol)
  {
    this->tol_ = new_tol;
  }

  void  LinSolverIterative::setMaxit(index_type new_maxit)
  {
    this->maxit_ = new_maxit;
  }

  void  LinSolverIterative::setRestart(index_type new_restart)
  {
    this->restart_ = new_restart;
  }

  void  LinSolverIterative::setConvCond(index_type new_conv_cond)
  {
    this->conv_cond_ = new_conv_cond;
  }

  void  LinSolverIterativeFGMRES::setFlexible(bool new_flex)
  {
    this->flexible_ = new_flex;
  }
}



