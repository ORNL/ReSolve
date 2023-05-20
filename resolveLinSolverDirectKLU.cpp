#include "resolveLinSolverDirectKLU.hpp"

namespace ReSolve {

  resolveLinSolverDirectKLU::resolveLinSolverDirectKLU()
  {
    Symbolic = nullptr;
    Numeric = nullptr;
    klu_defaults(&common) ;
  } 
  resolveLinSolverDirectKLU::~resolveLinSolverDirectKLU()
  {

  }
  void resolveLinSolverDirectKLU::setup(resolveMatrix* A)
  {
    this->A = A;
  }

  void resolveLinSolverDirectKLU::setupParameters(int ordering, double KLU_threshold, bool halt_if_singular) 
  {
    common.btf  = 0;
    common.ordering = ordering;
    common.tol = KLU_threshold;
    common.scale = -1;
    common.halt_if_singular = halt_if_singular;
  }

  int resolveLinSolverDirectKLU::analyze() 
  {
    Symbolic = klu_analyze(A->getNumRows(), A->getCsrRowPointers("cpu"), A->getCsrColIndices("cpu"), &common) ;
    if (Symbolic == nullptr){
      printf("Symbolic factorization crashed withcommon.status = %d \n", common.status);
      return -1;
    }
    return 0;
  }

  int resolveLinSolverDirectKLU::factorize() 
  {
    Numeric = klu_factor(A->getCsrRowPointers("cpu"), A->getCsrColIndices("cpu"),A->getCsrValues("cpu"), Symbolic, &common);
    if (Numeric == nullptr){
      return -1;
    }
    return 0;
  }

  int  resolveLinSolverDirectKLU::refactorize() 
  {
    int kluStatus = klu_refactor (A->getCsrRowPointers("cpu"), A->getCsrColIndices("cpu"), A->getCsrValues("cpu"), Symbolic, Numeric, &common);

    if (!kluStatus){
      //display error
      return -1;
    }
    return 0;
  }

  int resolveLinSolverDirectKLU::solve(resolveVector* rhs, resolveVector* x) 
  {
    //copy the vector

    //  std::memcpy(x, rhs, A->getNumRows() * sizeof(resolveReal));

    x->update(rhs->getData("cpu"), "cpu", "cpu");
    x->setDataUpdated("cpu");

    int kluStatus = klu_solve(Symbolic, Numeric, A->getNumRows(), 1, x->getData("cpu"), &common);

    if (!kluStatus){
      return -1;
    }

    return 0;
  }
}


