#include "tesolveLinSolverDirectKLU.hpp"

namespace ReSolve {

  resolveLinSolverDirectKLU::resolveLinSolverDirectKLU()
  {
    Symbolic = nullptr;
    Numeric = nullptr;
    klu_defaults(&Common) ;
  } 
  resolveLinSolverDirectKLU::~resolveLinSolverDirectKLU()
  {

  }
  void resolveLinSolverDirectKLU::setup(resolveMatrix* A)
  {
  }

  void resolveLinSolverDirectKLU::setupParameters(int ordering, double KLU_threshold, bool halt_if_singular) 
  {
    Common->btf  = 0;
    Common->ordering = ordering;
    Common->tol = KLU_threshold;
    Common->scale = -1;
    Common->halt_if_singular=halt_if_singular;
  }


  void resolveLinSolverDirectKLU::analyze() 
  {
    Symbolic = klu_analyze(A->getNumRows(), A->getiCsrRowPointers("cpu"), A->getCsrColumnIndices("cpu"), &Common) ;
    if (Symbolic == NULL){
      exit(1);
    }
  }
  void resolveLinSolverDirectKLU::factorize() 
  {

    Numeric = klu_factor(A->getCsrRowPointers("cpu"), A->getCsrColumnIndices("cpu"),A->getCsrVals("cpu"), Symbolic, &Common);



    void resolveLinSolverDirectKLU::refactorize() 
    {

    }
    resolveReal* resolveLinSolverDirectKLU::solve(resolveReal* rhs) 
    {

    }
  }

}
