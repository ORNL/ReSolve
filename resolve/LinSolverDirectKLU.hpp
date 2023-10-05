#pragma once
#include "klu.h"
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include "LinSolver.hpp"

namespace ReSolve 
{
  class LinSolverDirectKLU : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public:
      LinSolverDirectKLU();
      ~LinSolverDirectKLU();
      int setup(matrix::Sparse* A);
     
      void setupParameters(int ordering, double KLU_threshold, bool halt_if_singular);

      int analyze(); //the same as symbolic factorization
      int factorize();
      int refactorize();
      int solve(vector_type* rhs, vector_type* x); 
    
      matrix::Sparse* getLFactor(); 
      matrix::Sparse* getUFactor(); 
      index_type*  getPOrdering();
      index_type*  getQOrdering();

    private:
      klu_common Common_; //settings
      klu_symbolic* Symbolic_;
      klu_numeric* Numeric_; 
  };
}