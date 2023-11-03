#pragma once
#include <string>
#include "Common.hpp"

namespace ReSolve 
{
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  // Forward declaration of VectorHandler class
  class VectorHandler;

  // Forward declaration of matrix::Sparse class
  namespace matrix
  {
    class Sparse;
  }

  // Forward declaration of MatrixHandler class
  class MatrixHandler;

  class LinSolver 
  {
    protected:
      using vector_type = vector::Vector;

    public:
      LinSolver();
      virtual ~LinSolver();

      real_type evaluateResidual();
        
    protected:  
      matrix::Sparse* A_;
      real_type* rhs_;
      real_type* sol_;

      MatrixHandler* matrix_handler_;
      VectorHandler* vector_handler_;
  };

  class LinSolverDirect : public LinSolver 
  {
    public:
      LinSolverDirect();
      virtual ~LinSolverDirect();
      //return 0 if successful!
      virtual int setup(matrix::Sparse* A,
                        matrix::Sparse* L,
                        matrix::Sparse* U,
                        index_type*     P,
                        index_type*     Q,
                        vector_type*  rhs);
                        
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
      virtual int setup(matrix::Sparse* A);

      virtual int  solve(vector_type* rhs, vector_type* init_guess);
  };
}
