#pragma once
#include <string>
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"
#include "resolveVector.hpp"
#include "resolveMatrixHandler.hpp"
#include "resolveVectorHandler.hpp"

namespace ReSolve 
{
  class resolveLinSolver 
  {
    public:
      resolveLinSolver();
      ~resolveLinSolver();

      virtual void setup(resolveMatrix* A);
      resolveReal evaluateResidual();
        
    protected:  
      resolveMatrix* A_;
      resolveReal* rhs_;
      resolveReal* sol_;

      resolveMatrixHandler *matrix_handler_;
      resolveVectorHandler *vector_handler_;
  };

  class resolveLinSolverDirect : public resolveLinSolver 
  {
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
      resolveMatrix* L_;
      resolveMatrix* U_;
      resolveInt* P_;
      resolveInt* Q_;
      bool factors_extracted_;
  };

  class resolveLinSolverIterative : public resolveLinSolver 
  {
    public:
      resolveLinSolverIterative();
      ~resolveLinSolverIterative();

      virtual resolveReal* solve(resolveVector* rhs, resolveVector* init_guess);

  };
}
