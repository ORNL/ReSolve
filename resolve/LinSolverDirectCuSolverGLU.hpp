#pragma once
#include "Common.hpp"
#include "Matrix.hpp"
#include "LinSolver.hpp"
#include "cusolver_defs.hpp"

namespace ReSolve 
{
  class LinSolverDirectCuSolverGLU : public LinSolverDirect 
  {
    public:
      LinSolverDirectCuSolverGLU(LinAlgWorkspace* workspace);
      ~LinSolverDirectCuSolverGLU();

      int refactorize();
      int solve(Vector* rhs, Vector* x);

      int setup(Matrix* A, Matrix* L, Matrix* U, Int* P, Int* Q);
    
    private:
      void addFactors(Matrix* L, Matrix* U); //create L+U from sepeate L, U factors
      Matrix* M_;//the matrix that contains added factors
      //note: we need cuSolver handle, we can copy it from the workspace to avoid double allocation
      cusparseMatDescr_t descr_M_; //this is NOT sparse matrix descriptor
      cusparseMatDescr_t descr_A_; //this is NOT sparse matrix descriptor
      LinAlgWorkspace *workspace_;// so we can copy cusparse handle
      cusolverSpHandle_t handle_cusolversp_; 
      cusolverStatus_t status_cusolver_;
      cusparseStatus_t status_cusparse_;
      csrgluInfo_t info_M_;
      void* glu_buffer_;
      double r_nrminf_;
      int ite_refine_succ_; 
  };
}
