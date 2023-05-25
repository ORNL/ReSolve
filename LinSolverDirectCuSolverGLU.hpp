#pragma once
#include "Common.hpp"
#include "Matrix.hpp"
#include "LinSolver.hpp"
#include "cusolver_defs.hpp"

namespace ReSolve 
{
  class resolveLinSolverDirectCuSolverGLU : public resolveLinSolverDirect 
  {
    public:
      resolveLinSolverDirectCuSolverGLU(resolveLinAlgWorkspace* workspace);
      ~resolveLinSolverDirectCuSolverGLU();

      int refactorize();
      int solve(resolveVector* rhs, resolveVector* x);

      void setup(resolveMatrix* A, resolveMatrix* L, resolveMatrix* U, resolveInt* P, resolveInt* Q);
    
    private:
      void addFactors(resolveMatrix* L, resolveMatrix* U); //create L+U from sepeate L, U factors
      resolveMatrix* M_;//the matrix that contains added factors
      //note: we need cuSolver handle, we can copy it from the workspace to avoid double allocation
      cusparseMatDescr_t descr_M_; //this is NOT sparse matrix descriptor
      cusparseMatDescr_t descr_A_; //this is NOT sparse matrix descriptor
      resolveLinAlgWorkspace *workspace_;// so we can copy cusparse handle
      cusolverSpHandle_t handle_cusolversp_; 
      cusolverStatus_t status_cusolver_;
      csrgluInfo_t info_M_;
      void* glu_buffer_;
      double r_nrminf_;
      int ite_refine_succ_; 
  };
}
