/**
 * @file CholeskySolverCuDssCuda.cpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief CUDA cuDSS implementation of Cholesky Solver
 */

#include "CholeskySolverCuDssCuda.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    CholeskySolverCuDssCuda::CholeskySolverCuDssCuda()
    {
      cudssCreate(&cudss_handle_);
      cudssConfigCreate(&cudss_config_);
      cudssDataCreate(cudss_handle_, &cudss_data_);
    }

    CholeskySolverCuDssCuda::~CholeskySolverCuDssCuda()
    {
      cudssDataDestroy(cudss_handle_, cudss_data_);
      cudssConfigDestroy(cudss_config_);
      cudssDestroy(cudss_handle_);
      cudssMatrixDestroy(descr_A_cudss_);
      cudssMatrixDestroy(descr_b_);
      cudssMatrixDestroy(descr_x_);
    }

    void CholeskySolverCuDssCuda::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;

      cudssMatrixCreateCsr(&descr_A_cudss_,
                           A_->getNumRows(),
                           A_->getNumColumns(),
                           A_->getNnz(),
                           A_->getRowData(memory::DEVICE),
                           nullptr, // Row end offsets (null for standard CSR)
                           A_->getColData(memory::DEVICE),
                           A_->getValues(memory::DEVICE),
                           CUDA_R_32I,
                           CUDA_R_64F,
                           CUDSS_MTYPE_SPD,
                           CUDSS_MVIEW_LOWER,
                           CUDSS_BASE_ZERO);
    }

    /**
     * @brief Perform symbolic analysis for the Cholesky factorization
     */
    void CholeskySolverCuDssCuda::symbolicAnalysis()
    {
      cudssMatrixCreateDn(&descr_b_,
                          A_->getNumRows(),
                          1,
                          A_->getNumRows(),
                          nullptr,
                          CUDA_R_64F,
                          CUDSS_LAYOUT_COL_MAJOR);
      cudssMatrixCreateDn(&descr_x_,
                          A_->getNumRows(),
                          1,
                          A_->getNumRows(),
                          nullptr,
                          CUDA_R_64F,
                          CUDSS_LAYOUT_COL_MAJOR);
      cudssExecute(cudss_handle_,
                   CUDSS_PHASE_ANALYSIS,
                   cudss_config_,
                   cudss_data_,
                   descr_A_cudss_,
                   descr_x_,
                   descr_b_);
    }

    /**
     * @brief Perform numerical factorization for the Cholesky factorization
     *
     * @param[in] tol - Tolerance for zero pivot detection.
     */
    void CholeskySolverCuDssCuda::numericalFactorization(real_type tol)
    {
      cudssConfigSet(cudss_config_, CUDSS_CONFIG_PIVOT_EPSILON, &tol, sizeof(real_type));
      cudssStatus_t status = cudssExecute(cudss_handle_,
                                          CUDSS_PHASE_FACTORIZATION,
                                          cudss_config_,
                                          cudss_data_,
                                          descr_A_cudss_,
                                          descr_x_,
                                          descr_b_);
      if (status != CUDSS_STATUS_SUCCESS)
      {
        out::error() << "Cholesky factorization failed with status: " << status << "\n";
      }
    }

    /**
     * @brief Solve the linear system Ax = b
     *
     * Uses the `cusolverSpDcsrcholSolve` routine.
     *
     * @param[out] x - Solution vector.
     * @param[in]  b - Right-hand side vector.
     */
    void CholeskySolverCuDssCuda::solve(vector::Vector* x, vector::Vector* b)
    {
      if (descr_b_)
      {
        cudssMatrixDestroy(descr_b_);
      }
      if (descr_x_)
      {
        cudssMatrixDestroy(descr_x_);
      }

      cudssMatrixCreateDn(&descr_b_,
                          b->getSize(),
                          b->getNumVectors(),
                          b->getSize(),
                          b->getData(memory::DEVICE),
                          CUDA_R_64F,
                          CUDSS_LAYOUT_COL_MAJOR);
      cudssMatrixCreateDn(&descr_x_,
                          b->getSize(),
                          b->getNumVectors(),
                          b->getSize(),
                          x->getData(memory::DEVICE),
                          CUDA_R_64F,
                          CUDSS_LAYOUT_COL_MAJOR);

      cudssStatus_t status = cudssExecute(cudss_handle_, CUDSS_PHASE_SOLVE, cudss_config_, cudss_data_, descr_A_cudss_, descr_x_, descr_b_);
      if (status != CUDSS_STATUS_SUCCESS)
      {
        out::error() << "cuDSS triangular solve failed with status: " << status << "\n";
      }
    }
  } // namespace hykkt
} // namespace ReSolve
