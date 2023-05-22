#pragma once
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"
#include "resolveLinSolver.hpp"
#include "cusolverRf.h"


namespace ReSolve {
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
      cusolverRfHandle_t handle_cusolverrf;
      cusolverStatus_t status_cusolverrf;
      
      resolveInt* d_P;
      resolveInt* d_Q;
      resolveReal* d_T;
  };
}
