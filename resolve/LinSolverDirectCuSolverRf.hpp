#pragma once
#include "Common.hpp"
#include "Matrix.hpp"
#include "LinSolver.hpp"
#include "cusolverRf.h"

namespace ReSolve 
{
  class LinSolverDirectCuSolverRf : public LinSolverDirect 
  {
    public: 
      LinSolverDirectCuSolverRf();
      ~LinSolverDirectCuSolverRf();
      
      int setup(Matrix* A, Matrix* L, Matrix* U, Int* P, Int* Q);

      void setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg);
      
      int refactorize();
      int solve(Vector* rhs, Vector* x);

    private:
      cusolverRfHandle_t handle_cusolverrf_;
      cusolverStatus_t status_cusolverrf_;
      
      Int* d_P_;
      Int* d_Q_;
      Real* d_T_;
  };
}
