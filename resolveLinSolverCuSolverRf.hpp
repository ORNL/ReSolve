
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"
#include "resolveLinSolver.hpp"

namespace ReSolve {
  class resolveLinSolverDirectRf : public resolveLinSolverDirect 
  {
    public: 
      resolveLinSolverDirectKLU();
      ~resolveLinSolverDirectKLU();
      
      void setup(resolveMatrix* A, resolveMatrix* L, resolveMatrix* U, resolveInt* P, resolveInt* Q);

      void setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg);
      
      int refactorize();
      int solve(resolveReal* rhs, resolveReal* x);
    private:
      cusolverRfHandle_t handle_cusolverrf;
  };
}
