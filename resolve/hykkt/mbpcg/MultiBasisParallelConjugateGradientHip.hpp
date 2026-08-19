#pragma once

#include "MultiBasisParallelConjugateGradientImpl.hpp"

#include "SpMMHip.hpp"

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    class MultiBasisParallelConjugateGradientHip : public MultiBasisParallelConjugateGradientImpl
    {
    public:
      MultiBasisParallelConjugateGradientHip(VectorHandler* vector_handler);
      ~MultiBasisParallelConjugateGradientHip();

      int setup(index_type k);
      int SpMM(matrix::Csr* A, vector::Vector* X, vector::Vector* result);
      int bestBasis(vector::Vector* R, index_type* h_best_basis, real_type* h_best_basis_norm);
      int qr(vector::Vector* Q, vector::Vector* R, memory::MemorySpace memspace);
      int updateXR(vector::Vector* Xi_inv, vector::Vector* Sigma, vector::Vector* S, vector::Vector* A_S, vector::Vector* Xi_Sigma, vector::Vector* X_res, vector::Vector* R_prec);
      int choleskyFactorizeSolve(vector::Vector* A, vector::Vector* B, vector::Vector* X);
      int updateP(vector::Vector* P, vector::Vector* R, vector::Vector* Xi_inv_chol, vector::Vector* Delta, memory::MemorySpace memspace);
      int multTSMTTSM(vector::Vector* A, vector::Vector* B, vector::Vector* C, memory::MemorySpace memspace);
      int updateSSigma(vector::Vector* W, vector::Vector* S, vector::Vector* Zeta, vector::Vector* Sigma, memory::MemorySpace memspace);
      int preconditionDense(vector::Vector* A, vector::Vector* d);

    private:
      MemoryHandler mem_;
      VectorHandler* vector_handler_;

      SpMMHip spmm_hypre_;

      matrix::Csr* A_; // pointer to the input matrix

      index_type num_sms_;
      index_type num_threads_;

      void(*cholesky_qr_kernel_)(real_type*, real_type*, index_type){nullptr};

      size_t best_basis_workspace_size_;
      void* d_best_basis_workspace_{nullptr};
      void* d_best_basis_result_{nullptr}; // Type is cub::KeyValuePair<index_type, real_type>, but cub can't be included in .hpp
      void* h_best_basis_result_{nullptr};
      index_type* d_best_basis_{nullptr};
      real_type* d_sq_norms_{nullptr};
    };
  } // namespace hykkt
} // namespace ReSolve
