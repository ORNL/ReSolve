
#pragma once
#include "Common.hpp"
#include "LinSolver.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include <rocsparse/rocsparse.h>
//#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hip/hip_runtime.h>
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

  class LinSolverDirectRocSparseILU0 : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public: 
      LinSolverDirectRocSparseILU0(LinAlgWorkspaceHIP* workspace);
      ~LinSolverDirectRocSparseILU0();
      
      int setup(matrix::Sparse* A,
                matrix::Sparse* L = nullptr,
                matrix::Sparse* U = nullptr,
                index_type*     P = nullptr,
                index_type*     Q = nullptr,
                vector_type* rhs  = nullptr);
      // if values of A change, but the nnz pattern does not, redo the analysis only (reuse buffers though)
      int reset(matrix::Sparse* A);
       
      int solve(vector_type* rhs, vector_type* x);
      int solve(vector_type* rhs);// the solutuon is returned IN RHS (rhs is overwritten)
    

    private:
      rocsparse_status status_rocsparse_;

      MemoryHandler mem_; ///< Device memory manager object
      LinAlgWorkspaceHIP* workspace_; 

      rocsparse_mat_descr descr_A_{nullptr};
      rocsparse_mat_descr descr_L_{nullptr};
      rocsparse_mat_descr descr_U_{nullptr};

      rocsparse_mat_info  info_A_{nullptr};
      
      void* buffer_; 
      
      real_type* d_aux1_;
      // since ILU OVERWRITES THE MATRIX values, we need a buffer to keep the values of ILU decomposition. 
      real_type* d_ILU_vals_;
  };
}// namespace
