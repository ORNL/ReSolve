#pragma once
#include <string>
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"

namespace ReSolve {
  class resolveLinSolver {
    public:
      resolveLinSolver();
      ~resolveLinSolver();

      virtual setup(resolveMatrix* A);
      resolveReal evaluteResidual();
    
      
    private:
      
      resolveMatrix* A;
      resolveReal* rhs;
      resolveReal* sol;

      resolveMatrixHandler *matrix_handler;
      resolveVectorHandler *vector_handler;
  };

  class resolveLinSolverDirect : resolveLinSolver {
    public:
      resolveLinSolverDirect();
      ~resolveLinSolverDirect();

      virtual void analyze(); //the same as symbolic factorization
      virtual void factorize();
      virtual void refactorize();
      virtual resolveReal* solve(resolveReal* rhs); 
     
      virtual resolvematrix* getLFactor(); 
      virtual resolvematrix* getUFactor(); 
      virtual resolveInt*  getPOrdering();
      virtual resolveInt*  getQOrdering();
  };

  class resolveLinSolverIterative : resolveLinSolver {
    public:
      resolveLinSolverIterative();
      ~resolveLinSolverIterative();

      virtual resolveReal* solve(resolveReal* rhs, resolveReal* init_guess);

  };
}
