#pragma once
#include "Common.hpp"
#include "LinSolver.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hip/hip_runtime.h>
#include <roctracer/roctx.h>

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

  
  class LinSolverDirectRocSolverRf : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public: 
      LinSolverDirectRocSolverRf(LinAlgWorkspaceHIP* workspace);
      ~LinSolverDirectRocSolverRf();
      
      int setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q, vector_type* rhs);
       
      int refactorize();
      int solve(vector_type* rhs, vector_type* x);
      int solve(vector_type* rhs);// the solutuon is returned IN RHS (rhs is overwritten)
    
      int setSolveMode(int mode); // should probably be enum 
      int getSolveMode(); //should be enum too

    private:
      rocblas_status status_rocblas_;
      
      index_type* d_P_;
      index_type* d_Q_;
      real_type* d_T_;

      MemoryHandler mem_; ///< Device memory manager object
      LinAlgWorkspaceHIP* workspace_; 

      // to be exported to matrix handler in a later time
      void addFactors(matrix::Sparse* L, matrix::Sparse* U); //create L+U from sepeate L, U factors
      rocsolver_rfinfo infoM_;
      matrix::Sparse* M_;//the matrix that contains added factors
      int solve_mode_;
  };
}
