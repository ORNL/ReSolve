/**
 * @file CholeskySolverCuDssCuda.hpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief Header for CUDA cuDSS implementation of Cholesky Solver using
 */

#pragma once

#include <cudss.h>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    class CholeskySolverCuDssCuda
    {
    public:
      CholeskySolverCuDssCuda();
      ~CholeskySolverCuDssCuda();

      void addMatrixInfo(matrix::Csr* A);
      void symbolicAnalysis();
      void numericalFactorization(real_type tol);
      void solve(vector::Vector* x, vector::Vector* b);

    private:
      MemoryHandler mem_;

      matrix::Csr* A_; // pointer to the input matrix

      cudssHandle_t cudss_handle_;
      cudssConfig_t cudss_config_;
      cudssData_t   cudss_data_;
      cudssMatrix_t descr_A_cudss_;
      cudssMatrix_t descr_b_;
      cudssMatrix_t descr_x_;
    };
  } // namespace hykkt
} // namespace ReSolve
