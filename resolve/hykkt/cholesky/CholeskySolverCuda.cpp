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
    CholeskySolverCuda::CholeskySolverCuda()
    {
      cusolverSpCreate(&cusolverHandle_);
      cusparseCreateMatDescr(&descrA_);
      cusolverSpCreateCsrcholInfo(&factorizationInfo_);
      buffer_ = nullptr;
    }

    CholeskySolverCuda::~CholeskySolverCuda()
    {
      cusolverSpDestroy(cusolverHandle_);
      cusparseDestroyMatDescr(descrA_);
      cusolverSpDestroyCsrcholInfo(factorizationInfo_);
      mem_.deleteOnDevice(buffer_);
    }

    void CholeskySolverCuda::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
    }

    /**
     * @brief Perform symbolic analysis for the Cholesky factorization
     *
     * Uses the `cusolverSpXcsrcholAnalysis` routine.
     */
    void CholeskySolverCuda::symbolicAnalysis()
    {
      cusolverSpXcsrcholAnalysis(cusolverHandle_,
                                 A_->getNumRows(),
                                 A_->getNnz(),
                                 descrA_,
                                 A_->getRowData(memory::DEVICE),
                                 A_->getColData(memory::DEVICE),
                                 factorizationInfo_);
      // Calculate size of buffer needed
      size_t internalDataBytes = 0;
      size_t workspaceBytes    = 0;
      cusolverSpDcsrcholBufferInfo(cusolverHandle_,
                                   A_->getNumRows(),
                                   A_->getNnz(),
                                   descrA_,
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

    /**
     * @brief Perform numerical factorization for the Cholesky factorization
     *
     * Uses the `cusolverSpDcsrcholFactor` routine.
     *
     * @param[in] tol - Tolerance for zero pivot detection.
     */
    void CholeskySolverCuda::numericalFactorization(real_type tol)
    {
      int singularity = 0;
      cusolverSpDcsrcholFactor(cusolverHandle_,
                               A_->getNumRows(),
                               A_->getNnz(),
                               descrA_,
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
      cusolverSpDcsrcholSolve(cusolverHandle_,
                              A_->getNumRows(),
                              b->getData(memory::DEVICE),
                              x->getData(memory::DEVICE),
                              factorizationInfo_,
                              buffer_);
      x->setDataUpdated(memory::DEVICE);
    }
  } // namespace hykkt
} // namespace ReSolve
