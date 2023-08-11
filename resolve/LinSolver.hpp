#pragma once
#include <string>
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>

namespace ReSolve 
{
  class LinSolver 
  {
    protected:
      using vector_type = vector::Vector;

    public:
      LinSolver();
      ~LinSolver();

      virtual int setup(matrix::Sparse* A);
      real_type evaluateResidual();
        
    protected:  
      matrix::Sparse* A_;
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
      virtual int solve(vector_type* rhs, vector_type* x); 
     
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

      virtual int  solve(vector_type* rhs, vector_type* init_guess);

  };
}
