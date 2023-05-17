#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"
#include "resolveLinSolver.hpp"

namespace ReSolve {
  class resolveLinSolverCuSolverGLU : public resolveLinSolverDirect 
  {
    public:
      resolveLinSolverCuSolverGLU();
      ~resolveLinSolverCuSolverGLU();

      int refactorize();
      int solve(resolveReal* rhs, resolveReal* x);

      void createM(); //create L+U from sepeate L, U factors
    private:
      resolveMatrix* M;
      //note: we need cuSolver handle, we can copy it from the workspace to avoid double allocation
      cusparseMatDescr_t descrM; //this is NOT sparse matrix descriptor
      cusparseMatDescr_t descrA; //this is NOT sparse matrix descriptor
  }
