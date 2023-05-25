#include "LinSolver.hpp"


namespace ReSolve 
{
  resolveLinSolver::resolveLinSolver()
  {
  }

  resolveLinSolver::~resolveLinSolver()
  {
    //destroy the matrix and hadlers
  }

  void resolveLinSolver::setup(resolveMatrix* A)
  {
    this->A_ = A;
  }

  resolveReal resolveLinSolver::evaluateResidual()
  {
    //to be implemented
    return 1.0;
  }

  resolveLinSolverDirect::resolveLinSolverDirect()
  {
    L_ = nullptr;
    U_ = nullptr;
    P_ = nullptr;
    Q_ = nullptr;
    factors_extracted_ = false;
  }

  resolveLinSolverDirect::~resolveLinSolverDirect()
  {
    delete L_;
    delete U_;
    delete [] P_;
    delete [] Q_;
  }

  int resolveLinSolverDirect::analyze()
  {
    return 0;
  } //the same as symbolic factorization

  int resolveLinSolverDirect::factorize()
  {
    factors_extracted_ = false;
    return 0;
  }

  int resolveLinSolverDirect::refactorize()
  {
    factors_extracted_ = false;
    return 0;
  }

  int resolveLinSolverDirect::solve(resolveVector* rhs, resolveVector* x) 
  {
    return 0;
  }

  resolveMatrix* resolveLinSolverDirect::getLFactor()
  {
    return nullptr;
  } 
  
  resolveMatrix* resolveLinSolverDirect::getUFactor()
  {
    return nullptr;
  } 
  
  resolveInt*  resolveLinSolverDirect::getPOrdering()
  {
    return nullptr;
  } 
  
  resolveInt*  resolveLinSolverDirect::getQOrdering()
  {
    return nullptr;
  } 
}



