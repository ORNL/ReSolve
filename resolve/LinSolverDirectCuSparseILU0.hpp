#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>

#include "Common.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/LinSolverDirect.hpp>

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
                vector_type* rhs  = nullptr) override;
      // if values of A change, but the nnz pattern does not, redo the analysis only (reuse buffers though)
      int reset(matrix::Sparse* A);
       
      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* rhs) override;
    

    private:
      cusparseStatus_t status_cusparse_;

      MemoryHandler mem_; ///< Device memory manager object
      LinAlgWorkspaceCUDA* workspace_{nullptr}; 

      cusparseMatDescr_t descr_A_{nullptr};
      
      cusparseSpMatDescr_t mat_L_{nullptr}; 
      cusparseSpMatDescr_t mat_U_{nullptr}; 
      
      cusparseSpSVDescr_t descr_spsv_L_{nullptr};
      cusparseSpSVDescr_t descr_spsv_U_{nullptr};
      csrilu02Info_t  info_A_{nullptr};
      
      void* buffer_{nullptr}; 
      void* buffer_L_{nullptr}; 
      void* buffer_U_{nullptr}; 
      
      real_type* d_aux1_{nullptr};
      real_type* d_aux2_{nullptr};
      
      cusparseDnVecDescr_t vec_X_{nullptr};
      cusparseDnVecDescr_t vec_Y_{nullptr};

      // since ILU OVERWRITES THE MATRIX values, we need a buffer to keep the values of ILU decomposition. 
      real_type* d_ILU_vals_{nullptr};
  };
}// namespace
