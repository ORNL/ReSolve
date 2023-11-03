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

  int LinSolverDirect::setup(matrix::Sparse* A,
                             matrix::Sparse* /* L */,
                             matrix::Sparse* /* U */,
                             index_type*     /* P */,
                             index_type*     /* Q */,
                             vector_type*  /* rhs */)
  {
    this->A_ = A;
    return 0;
  }

  int LinSolverDirect::analyze()
  {
    return 0;
  } //the same as symbolic factorization

  int LinSolverDirect::factorize()
  {
    factors_extracted_ = false;
    return 0;
  }

  int LinSolverDirect::refactorize()
  {
    factors_extracted_ = false;
    return 0;
  }

  int LinSolverDirect::solve(vector_type* /* rhs */, vector_type* /* x */) 
  {
    return 0;
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

  LinSolverIterative::LinSolverIterative()
  {
  }
  
  LinSolverIterative::~LinSolverIterative()
  {
  }

  int LinSolverIterative::setup(matrix::Sparse* A)
  {
    this->A_ = A;
    return 0;
  }

  int LinSolverIterative::solve(vector_type* /* rhs */, vector_type* /* init_guess */)
  {
    return 0;
  }
}



