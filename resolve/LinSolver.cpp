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

  int LinSolver::setup(Matrix* A)
  {
    this->A_ = A;
    return 0;
  }

  Real LinSolver::evaluateResidual()
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

  int LinSolverDirect::solve(Vector* rhs, Vector* x) 
  {
    return 0;
  }

  Matrix* LinSolverDirect::getLFactor()
  {
    return nullptr;
  } 
  
  Matrix* LinSolverDirect::getUFactor()
  {
    return nullptr;
  } 
  
  Int*  LinSolverDirect::getPOrdering()
  {
    return nullptr;
  } 
  
  Int*  LinSolverDirect::getQOrdering()
  {
    return nullptr;
  } 

  LinSolverIterative::LinSolverIterative()
  {
  }
  
  LinSolverIterative::~LinSolverIterative()
  {
  }


  int LinSolverIterative::solve(Vector* rhs, Vector* init_guess)
  {
    return 0;
  }
}



