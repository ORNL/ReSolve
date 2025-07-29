#include "CholeskySolverCuda.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;

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

    void CholeskySolverCuda::symbolicAnalysis()
    {
      cusolverSpXcsrcholAnalysis(cusolverHandle_,
                                 A_->getNumRows(),
                                 A_->getNnz(),
                                 descrA_,
                                 A_->getRowData(memory::DEVICE),
                                 A_->getColData(memory::DEVICE),
                                 factorizationInfo_);
      // TODO: Do we have a ReSolve type for size_t?
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
      mem_.allocateBufferOnDevice(&buffer_, workspaceBytes);
    }

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
        // TODO: What to do if A_ is singular?
      }
    }

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
