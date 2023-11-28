#pragma once
#include "Common.hpp"
#include "LinSolver.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include <cusparse.h>
#include <cuda_runtime.h>

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

  class LinSolverDirectCuSparseILU0 : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public: 
      LinSolverDirectCuSparseILU0(LinAlgWorkspaceCUDA* workspace);
      ~LinSolverDirectCuSparseILU0();
      
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
      cusparseStatus_t status_cusparse_;

      MemoryHandler mem_; ///< Device memory manager object
      LinAlgWorkspaceCUDA* workspace_; 

      cusparseMatDescr_t descr_A_{nullptr};
      
      cusparseSpMatDescr_t mat_L_; 
      cusparseSpMatDescr_t mat_U_; 
      
      cusparseSpSVDescr_t descr_spsv_L_{nullptr};
      cusparseSpSVDescr_t descr_spsv_U_{nullptr};
      csrilu02Info_t  info_A_{nullptr};
      
      void* buffer_; 
      void* buffer_L_; 
      void* buffer_U_; 
      
      real_type* d_aux1_;
      real_type* d_aux2_;
      
      cusparseDnVecDescr_t vec_X_;
      cusparseDnVecDescr_t vec_Y_;

      // since ILU OVERWRITES THE MATRIX values, we need a buffer to keep the values of ILU decomposition. 
      real_type* d_ILU_vals_;
  };
}// namespace
