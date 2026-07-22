/**
 * @file CholeskySolverCuda.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Header for CUDA implementation of Cholesky Solver using cuDSS
 */

#pragma once

#include <cublas_v2.h>

#ifdef RESOLVE_USE_CUDSS
#include <cudss.h>
#endif

#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include "CholeskySolverImpl.hpp"

namespace ReSolve
{
  namespace hykkt
  {
    class CholeskySolverCuda : public CholeskySolverImpl
    {
    public:
      CholeskySolverCuda(bool use_cudss = true);
      ~CholeskySolverCuda();

      void addMatrixInfo(matrix::Csr* A);
      void symbolicAnalysis();
      void numericalFactorization(real_type tol);
      void solve(vector::Vector* x, vector::Vector* b);

    private:
      MemoryHandler mem_;

      matrix::Csr* A_; // pointer to the input matrix

      bool use_cudss_;

#ifdef RESOLVE_USE_CUDSS
      cudssHandle_t cudss_handle_;
      cudssConfig_t cudss_config_;
      cudssData_t   cudss_data_;
      cudssMatrix_t descr_A_cudss_;
      cudssMatrix_t descr_b_;
      cudssMatrix_t descr_x_;
#endif

      // handle to the cuSPARSE library context
      cusolverSpHandle_t cusolverHandle_;
      cusparseMatDescr_t descr_A_cusolver_;  // descriptor for matrix A
      csrcholInfo_t      factorizationInfo_; // stores Cholesky factorization
      void*              buffer_;            // buffer for Cholesky factorization
    };
  } // namespace hykkt
} // namespace ReSolve
