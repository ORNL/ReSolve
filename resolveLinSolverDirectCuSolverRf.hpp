#pragma once
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"
#include "resolveLinSolver.hpp"
#include "cusolverRf.h"

namespace ReSolve 
{
  class resolveLinSolverDirectCuSolverRf : public resolveLinSolverDirect 
  {
    public: 
      resolveLinSolverDirectCuSolverRf();
      ~resolveLinSolverDirectCuSolverRf();
      
      void setup(resolveMatrix* A, resolveMatrix* L, resolveMatrix* U, resolveInt* P, resolveInt* Q);

      void setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg);
      
      int refactorize();
      int solve(resolveVector* rhs, resolveVector* x);

    private:
      cusolverRfHandle_t handle_cusolverrf_;
      cusolverStatus_t status_cusolverrf_;
      
      resolveInt* d_P_;
      resolveInt* d_Q_;
      resolveReal* d_T_;
  };
}
