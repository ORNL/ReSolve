#pragma once
#include "Common.hpp"
#include "Matrix.hpp"
#include "LinSolver.hpp"

namespace ReSolve 
{
  class LinSolverIterativeFGMRES : public LinSolverIterative
  {
    public:
      LinSolverIterativeFGMRES();
      LinSolverIterativeFGMRES(Int restart, Real tol, Int maxit);
      ~LinSolverIterativeFGMRES();

      int solve(Vector* rhs, Vector* x);
    private:
      //remember matrix handler and vector handler are inherited.
      LinAlgWorkspace* workspace_;
      
      Real tol_;
      Int maxit_;
      Int restart_;
      
      Real* V;


  };

}
