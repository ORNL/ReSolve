#pragma once
#include <string>
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"
#include "resolveVector.hpp"
#include "resolveMatrixHandler.hpp"
#include "resolveVectorHandler.hpp"
namespace ReSolve {
  class resolveLinSolver {
    public:
      resolveLinSolver();
      ~resolveLinSolver();

      virtual void setup(resolveMatrix* A);
      resolveReal evaluateResidual();
    
      
    protected:
      
      resolveMatrix* A;
      resolveReal* rhs;
      resolveReal* sol;

      resolveMatrixHandler *matrix_handler;
      resolveVectorHandler *vector_handler;
  };

  class resolveLinSolverDirect : public resolveLinSolver {
    public:
      resolveLinSolverDirect();
      ~resolveLinSolverDirect();
      //return 0 if successful!
      virtual int analyze(); //the same as symbolic factorization
      virtual int factorize();
      virtual int refactorize();
      virtual int solve(resolveVector* rhs, resolveVector* x); 
     
      virtual resolveMatrix* getLFactor(); 
      virtual resolveMatrix* getUFactor(); 
      virtual resolveInt*  getPOrdering();
      virtual resolveInt*  getQOrdering();
    protected:
      resolveMatrix* L;
      resolveMatrix* U;
      resolveInt* P;
      resolveInt* Q;
  };

  class resolveLinSolverIterative : public resolveLinSolver {
    public:
      resolveLinSolverIterative();
      ~resolveLinSolverIterative();

      virtual resolveReal* solve(resolveVector* rhs, resolveVector* init_guess);

  };
}
