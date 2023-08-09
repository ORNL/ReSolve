#pragma once
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include "LinSolver.hpp"
#include "cusolverRf.h"

namespace ReSolve 
{
  class LinSolverDirectCuSolverRf : public LinSolverDirect 
  {
    public: 
      LinSolverDirectCuSolverRf();
      ~LinSolverDirectCuSolverRf();
      
      int setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q);

      void setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg);
      
      int refactorize();
      int solve(Vector* rhs, Vector* x);
      int solve(Vector* rhs); ///< The solutuon is returned IN RHS (the rhs is overwritten)

    private:
      cusolverRfHandle_t handle_cusolverrf_;
      cusolverStatus_t status_cusolverrf_;
      
      index_type* d_P_;
      index_type* d_Q_;
      real_type* d_T_;
  };
}
