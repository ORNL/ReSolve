#pragma once
#include <string>
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include "Vector.hpp"
#include <resolve/matrix/MatrixHandler.hpp>
#include "VectorHandler.hpp"

namespace ReSolve 
{
  class LinSolver 
  {
    public:
      LinSolver();
      ~LinSolver();

      virtual int setup(matrix::Sparse* A);
      real_type evaluateResidual();
        
    protected:  
      matrix::Sparse* A_;
      real_type* rhs_;
      real_type* sol_;

      matrix::MatrixHandler *matrix_handler_;
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
     
      virtual matrix::Sparse* getLFactor(); 
      virtual matrix::Sparse* getUFactor(); 
      virtual index_type*  getPOrdering();
      virtual index_type*  getQOrdering();
    
    protected:
      matrix::Sparse* L_;
      matrix::Sparse* U_;
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
