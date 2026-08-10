#pragma once

#include "RandomizedConjugateGradientImpl.hpp"

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

namespace ReSolve
{
  namespace hykkt
  {
    class RandomizedConjugateGradientCuda : public RandomizedConjugateGradientImpl
    {
    public:
      RandomizedConjugateGradientCuda(VectorHandler* vector_handler);
      ~RandomizedConjugateGradientCuda();

      int setup(index_type k);
      int SpMMTallSkinny(matrix::Csr* A, vector::Vector* X, vector::Vector* result);
      int bestBasis(vector::Vector* R, index_type* h_best_basis, real_type* h_best_basis_norm);
      int choleskyQr(vector::Vector* W, vector::Vector* R, memory::MemorySpace memspace);
      int updateXRSplit(vector::Vector* Xi_inv, vector::Vector* Sigma, vector::Vector* S, vector::Vector* A_S, vector::Vector* Xi_Sigma, vector::Vector* X_res, vector::Vector* R_prec);
      int choleskyFactorizeSolve(vector::Vector* A, vector::Vector* B, vector::Vector* X);
      int updateW(vector::Vector* W, vector::Vector* L, vector::Vector* B, memory::MemorySpace memspace);
      int multTSMTTSM(vector::Vector* A, vector::Vector* B, vector::Vector* C, memory::MemorySpace memspace);
      int updateSSigma(vector::Vector* W, vector::Vector* S, vector::Vector* Zeta, vector::Vector* Sigma, memory::MemorySpace memspace);
      int preconditionDense(vector::Vector* A, vector::Vector* d);
      int hypreDevice_CSRMatrixMatvec(matrix::Csr* A, vector::Vector* X, vector::Vector* result);

    private:
      MemoryHandler mem_;
      VectorHandler* vector_handler_;

      matrix::Csr* A_; // pointer to the input matrix

      // handle to the cuSPARSE library context
      cusolverSpHandle_t cusolverHandle_;
      cusparseMatDescr_t descrA_;            // descriptor for matrix A
      csrcholInfo_t      factorizationInfo_; // stores Cholesky factorization
      void*              buffer_;            // buffer for Cholesky factorization

      index_type num_sms_;
      index_type num_threads_;
      index_type qr_blocks_per_sm_;

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
