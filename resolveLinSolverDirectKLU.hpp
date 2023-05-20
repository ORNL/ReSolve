
#pragma once
#include "klu.h"
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"
#include "resolveLinSolver.hpp"

namespace ReSolve {
  class resolveLinSolverDirectKLU : public resolveLinSolverDirect 
  {
    public:
      resolveLinSolverDirectKLU();
      ~resolveLinSolverDirectKLU();
      void setup(resolveMatrix* A);
     
      void setupParameters(int ordering, double KLU_threshold, bool halt_if_singular);


      int analyze(); //the same as symbolic factorization
      int factorize();
      int refactorize();
      int solve(resolveVector* rhs, resolveVector* x); 
    
    private:

      klu_common common; //settings
      klu_symbolic* Symbolic;
      klu_numeric* Numeric; 
  };
}
