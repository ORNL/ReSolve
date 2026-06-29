#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

namespace ReSolve
{
  namespace hykkt
  {
    class RandomizedConjugateGradientCuda
    {
    public:
      RandomizedConjugateGradientCuda();
      ~RandomizedConjugateGradientCuda();

      int SpMMTallSkinny(matrix::Csr* A, const vector::Vector* X, vector::Vector* result);

    private:
      MemoryHandler mem_;

      matrix::Csr* A_; // pointer to the input matrix

      // handle to the cuSPARSE library context
      cusolverSpHandle_t cusolverHandle_;
      cusparseMatDescr_t descrA_;            // descriptor for matrix A
      csrcholInfo_t      factorizationInfo_; // stores Cholesky factorization
      void*              buffer_;            // buffer for Cholesky factorization
    };
  } // namespace hykkt
} // namespace ReSolve
