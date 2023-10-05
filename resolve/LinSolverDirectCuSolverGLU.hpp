#pragma once
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include "LinSolver.hpp"
#include "cusolver_defs.hpp"

namespace ReSolve 
{
  class LinSolverDirectCuSolverGLU : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public:
      LinSolverDirectCuSolverGLU(LinAlgWorkspace* workspace);
      ~LinSolverDirectCuSolverGLU();

      int refactorize();
      int solve(vector_type* rhs, vector_type* x);

      int setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q);
    
    private:
      void addFactors(matrix::Sparse* L, matrix::Sparse* U); //create L+U from sepeate L, U factors
      matrix::Sparse* M_;//the matrix that contains added factors
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