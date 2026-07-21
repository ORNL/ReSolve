/**
 * @file CholeskySolverCuda.cpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief CUDA implementation of Cholesky Solver
 */

#include "CholeskySolverCuda.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    CholeskySolverCuda::CholeskySolverCuda(bool use_cudss)
      : use_cudss_(use_cudss)
    {
      if (use_cudss_)
      {
        cudssCreate(&cudss_handle_);
        cudssConfigCreate(&cudss_config_);
        cudssDataCreate(cudss_handle_, &cudss_data_);
      }
      else
      {
        cusolverSpCreate(&cusolverHandle_);
        cusparseCreateMatDescr(&descr_A_cusolver_);
        cusolverSpCreateCsrcholInfo(&factorizationInfo_);
        buffer_ = nullptr;
      }
    }

    CholeskySolverCuda::~CholeskySolverCuda()
    {
      if (use_cudss_)
      {
        cudssDataDestroy(cudss_handle_, cudss_data_);
        cudssConfigDestroy(cudss_config_);
        cudssDestroy(cudss_handle_);
        cudssMatrixDestroy(descr_A_cudss_);
        cudssMatrixDestroy(descr_b_);
        cudssMatrixDestroy(descr_x_);
      }
      else
      {
        cusolverSpDestroy(cusolverHandle_);
        cusparseDestroyMatDescr(descr_A_cusolver_);
        cusolverSpDestroyCsrcholInfo(factorizationInfo_);
        mem_.deleteOnDevice(buffer_);
      }
    }

    void CholeskySolverCuda::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;

      if (use_cudss_)
      {
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
    }

    /**
     * @brief Perform symbolic analysis for the Cholesky factorization
     */
    void CholeskySolverCuda::symbolicAnalysis()
    {
      if (use_cudss_)
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
      else
      {
        cusolverSpXcsrcholAnalysis(cusolverHandle_,
                            A_->getNumRows(),
                            A_->getNnz(),
                            descr_A_cusolver_,
                            A_->getRowData(memory::DEVICE),
                            A_->getColData(memory::DEVICE),
                            factorizationInfo_);
        // Calculate size of buffer needed
        size_t internalDataBytes = 0;
        size_t workspaceBytes    = 0;
        cusolverSpDcsrcholBufferInfo(cusolverHandle_,
                                    A_->getNumRows(),
                                    A_->getNnz(),
                                    descr_A_cusolver_,
                                    A_->getValues(memory::DEVICE),
                                    A_->getRowData(memory::DEVICE),
                                    A_->getColData(memory::DEVICE),
                                    factorizationInfo_,
                                    &internalDataBytes,
                                    &workspaceBytes);
        if (buffer_ != nullptr)
        {
          mem_.deleteOnDevice(buffer_);
        }
        mem_.allocateBufferOnDevice(&buffer_, workspaceBytes);
      }
    }

    /**
     * @brief Perform numerical factorization for the Cholesky factorization
     * 
     * @param[in] tol - Tolerance for zero pivot detection.
     */
    void CholeskySolverCuda::numericalFactorization(real_type tol)
    {
      if (use_cudss_)
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
      else
      {
        int singularity = 0;
        cusolverSpDcsrcholFactor(cusolverHandle_,
                                A_->getNumRows(),
                                A_->getNnz(),
                                descr_A_cusolver_,
                                A_->getValues(memory::DEVICE),
                                A_->getRowData(memory::DEVICE),
                                A_->getColData(memory::DEVICE),
                                factorizationInfo_,
                                buffer_);
        cusolverSpDcsrcholZeroPivot(cusolverHandle_,
                                    factorizationInfo_,
                                    tol,
                                    &singularity);
        if (singularity >= 0)
        {
          out::error() << "Cholesky factorization failed with singularity at index: " << singularity << "\n";
        }
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
    void CholeskySolverCuda::solve(vector::Vector* x, vector::Vector* b)
    {
      if (use_cudss_)
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
      else
      {
        for (index_type i = 0; i < b->getNumVectors(); i++)
        {
          cusolverSpDcsrcholSolve(cusolverHandle_,
                                  A_->getNumRows(),
                                  b->getData(i, memory::DEVICE),
                                  x->getData(i, memory::DEVICE),
                                  factorizationInfo_,
                                  buffer_);
        }
      }

      x->setDataUpdated(memory::DEVICE);
    }
  } // namespace hykkt
} // namespace ReSolve