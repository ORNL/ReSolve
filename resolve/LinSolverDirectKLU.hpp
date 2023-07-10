#pragma once
#include "klu.h"
#include "Common.hpp"
#include "Matrix.hpp"
#include "LinSolver.hpp"

namespace ReSolve 
{
  class LinSolverDirectKLU : public LinSolverDirect 
  {
    public:
      LinSolverDirectKLU();
      ~LinSolverDirectKLU();
      int setup(Matrix* A);
     
      void setupParameters(int ordering, double KLU_threshold, bool halt_if_singular);

      int analyze(); //the same as symbolic factorization
      int factorize();
      int refactorize();
      int solve(Vector* rhs, Vector* x); 
    
      Matrix* getLFactor(); 
      Matrix* getUFactor(); 
      Int*  getPOrdering();
      Int*  getQOrdering();

    private:
      klu_common Common_; //settings
      klu_symbolic* Symbolic_;
      klu_numeric* Numeric_; 
  };
}
