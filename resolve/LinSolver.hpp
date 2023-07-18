#pragma once
#include <string>
#include "Common.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "MatrixHandler.hpp"
#include "VectorHandler.hpp"

namespace ReSolve 
{
  class LinSolver 
  {
    public:
      LinSolver();
      ~LinSolver();

      virtual int setup(Matrix* A);
      real_type evaluateResidual();
        
    protected:  
      Matrix* A_;
      real_type* rhs_;
      real_type* sol_;

      MatrixHandler *matrix_handler_;
      VectorHandler *vector_handler_;
  };

  class LinSolverDirect : public LinSolver 
  {
    public:
      LinSolverDirect();
      ~LinSolverDirect();
      //return 0 if successful!
      virtual int analyze(); //the same as symbolic factorization
      virtual int factorize();
      virtual int refactorize();
      virtual int solve(Vector* rhs, Vector* x); 
     
      virtual Matrix* getLFactor(); 
      virtual Matrix* getUFactor(); 
      virtual index_type*  getPOrdering();
      virtual index_type*  getQOrdering();
    
    protected:
      Matrix* L_;
      Matrix* U_;
      index_type* P_;
      index_type* Q_;
      bool factors_extracted_;
  };

  class LinSolverIterative : public LinSolver 
  {
    public:
      LinSolverIterative();
      ~LinSolverIterative();

      virtual int  solve(Vector* rhs, Vector* init_guess);

  };
}
