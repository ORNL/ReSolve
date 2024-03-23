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
  
  class GramSchmidt;

  class LinSolver 
  {
    protected:
      using vector_type = vector::Vector;

    public:
      LinSolver();
      virtual ~LinSolver();

      real_type evaluateResidual();
        
    protected:  
      matrix::Sparse* A_{nullptr};
      real_type* rhs_{nullptr};
      real_type* sol_{nullptr};

      MatrixHandler* matrix_handler_{nullptr};
      VectorHandler* vector_handler_{nullptr};
  };

  class LinSolverDirect : public LinSolver 
  {
    public:
      LinSolverDirect();
      virtual ~LinSolverDirect();
      virtual int setup(matrix::Sparse* A = nullptr,
                        matrix::Sparse* L = nullptr,
                        matrix::Sparse* U = nullptr,
                        index_type*     P = nullptr,
                        index_type*     Q = nullptr,
                        vector_type*  rhs = nullptr);

      virtual int analyze(); //the same as symbolic factorization
      virtual int factorize();
      virtual int refactorize();
      virtual int solve(vector_type* rhs, vector_type* x) = 0;
      virtual int solve(vector_type* x) = 0;
     
      virtual matrix::Sparse* getLFactor(); 
      virtual matrix::Sparse* getUFactor(); 
      virtual index_type*  getPOrdering();
      virtual index_type*  getQOrdering();

      virtual void setPivotThreshold(real_type tol);
      virtual void setOrdering(int ordering);
      virtual void setHaltIfSingular(bool is_halt);

      virtual real_type getMatrixConditionNumber();
    
    protected:
      matrix::Sparse* L_{nullptr};
      matrix::Sparse* U_{nullptr};
      index_type* P_{nullptr};
      index_type* Q_{nullptr};

      int ordering_{1}; // 0 = AMD, 1 = COLAMD, 2 = user provided P, Q
      real_type pivot_threshold_tol_{0.1};
      bool halt_if_singular_{false};
  };

  class LinSolverIterative : public LinSolver 
  {
    public:
      LinSolverIterative();
      virtual ~LinSolverIterative();
      virtual int setup(matrix::Sparse* A);
      virtual int resetMatrix(matrix::Sparse* A) = 0;
      virtual int setupPreconditioner(std::string type, LinSolverDirect* LU_solver) = 0;

      virtual int  solve(vector_type* rhs, vector_type* init_guess) = 0;

      virtual real_type getFinalResidualNorm() const;
      virtual real_type getInitResidualNorm() const;
      virtual index_type getNumIter() const;

      virtual int setOrthogonalization(GramSchmidt* gs);

      real_type getTol();
      index_type getMaxit();
      index_type getRestart();
      index_type getConvCond();
      bool getFlexible();

      void setTol(real_type new_tol);
      void setMaxit(index_type new_maxit);
      virtual int setRestart(index_type new_restart) = 0;
      void setConvCond(index_type new_conv_cond);
      void setFlexible(bool new_flexible);

    protected:
      real_type initial_residual_norm_;
      real_type final_residual_norm_;
      index_type total_iters_;

      real_type tol_{1e-14};
      index_type maxit_{100};
      index_type restart_{10};
      index_type conv_cond_{0};
      bool flexible_{true}; // if can be run as "normal" GMRES if needed, set flexible_ to false. Default is true of course.
  };
}
