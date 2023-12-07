#pragma once
#include "Common.hpp"
#include "LinSolver.hpp"
#include "cusolverRf.h"
#include <resolve/MemoryUtils.hpp>

namespace ReSolve 
{
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  // Forward declaration of matrix::Sparse class
  namespace matrix
  {
    class Sparse;
  }

  class LinSolverDirectCuSolverRf : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public: 
      LinSolverDirectCuSolverRf();
      ~LinSolverDirectCuSolverRf();
      
      int setup(matrix::Sparse* A,
                matrix::Sparse* L,
                matrix::Sparse* U,
                index_type*     P,
                index_type*     Q,
                vector_type* rhs = nullptr) override;
      
      int refactorize() override;
      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* rhs) override; // rhs overwritten by solution

      void setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg);
      int setNumericalProperties(double nzero, double nboost);//these two NEED TO BE DOUBLE
    private:
      cusolverRfHandle_t handle_cusolverrf_;
      cusolverStatus_t status_cusolverrf_;
      
      index_type* d_P_{nullptr};
      index_type* d_Q_{nullptr};
      real_type* d_T_{nullptr};
      bool setup_completed_{false};
      
      MemoryHandler mem_; ///< Device memory manager object
  };
}
