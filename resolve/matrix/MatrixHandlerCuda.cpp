#include <algorithm>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#include "MatrixHandlerCuda.hpp"

namespace ReSolve {
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  MatrixHandlerCuda::~MatrixHandlerCuda()
  {
  }

  MatrixHandlerCuda::MatrixHandlerCuda(LinAlgWorkspaceCUDA* new_workspace)
  {
    workspace_ = new_workspace;
  }

  void MatrixHandlerCuda::setValuesChanged(bool values_changed)
  {
    values_changed_ = values_changed;
  }


  int MatrixHandlerCuda::matvec(matrix::Sparse* Ageneric, 
                                vector_type* vec_x, 
                                vector_type* vec_result, 
                                const real_type* alpha, 
                                const real_type* beta,
                                std::string matrixFormat) 
  {
    using namespace constants;
    int error_sum = 0;
    if (matrixFormat == "csr") {
      matrix::Csr* A = dynamic_cast<matrix::Csr*>(Ageneric);
      //result = alpha *A*x + beta * result
      cusparseStatus_t status;
      LinAlgWorkspaceCUDA* workspaceCUDA = workspace_;
      cusparseDnVecDescr_t vecx = workspaceCUDA->getVecX();
      cusparseCreateDnVec(&vecx, A->getNumRows(), vec_x->getData("cuda"), CUDA_R_64F);


      cusparseDnVecDescr_t vecAx = workspaceCUDA->getVecY();
      cusparseCreateDnVec(&vecAx, A->getNumRows(), vec_result->getData("cuda"), CUDA_R_64F);

      cusparseSpMatDescr_t matA = workspaceCUDA->getSpmvMatrixDescriptor();

      void* buffer_spmv = workspaceCUDA->getSpmvBuffer();
      cusparseHandle_t handle_cusparse = workspaceCUDA->getCusparseHandle();
      if (values_changed_) {
        status = cusparseCreateCsr(&matA, 
                                   A->getNumRows(),
                                   A->getNumColumns(),
                                   A->getNnzExpanded(),
                                   A->getRowData("cuda"),
                                   A->getColData("cuda"),
                                   A->getValues("cuda"), 
                                   CUSPARSE_INDEX_32I, 
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F);
        error_sum += status;
        values_changed_ = false;
      }
      if (!workspaceCUDA->matvecSetup()) {
        //setup first, allocate, etc.
        size_t bufferSize = 0;

        status = cusparseSpMV_bufferSize(handle_cusparse, 
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &MINUSONE,
                                         matA,
                                         vecx,
                                         &ONE,
                                         vecAx,
                                         CUDA_R_64F,
                                         CUSPARSE_SPMV_CSR_ALG2, 
                                         &bufferSize);
        error_sum += status;
        mem_.deviceSynchronize();
        mem_.allocateBufferOnDevice(&buffer_spmv, bufferSize);
        workspaceCUDA->setSpmvMatrixDescriptor(matA);
        workspaceCUDA->setSpmvBuffer(buffer_spmv);

        workspaceCUDA->matvecSetupDone();
      } 

      status = cusparseSpMV(handle_cusparse,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,       
                            alpha, 
                            matA, 
                            vecx, 
                            beta, 
                            vecAx, 
                            CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG2, 
                            buffer_spmv);
      error_sum += status;
      mem_.deviceSynchronize();
      if (status)
        out::error() << "Matvec status: " << status 
                      << "Last error code: " << mem_.getLastDeviceError() << std::endl;
      vec_result->setDataUpdated("cuda");

      cusparseDestroyDnVec(vecx);
      cusparseDestroyDnVec(vecAx);
      return error_sum;
    } else {
      out::error() << "MatVec not implemented (yet) for " 
                   << matrixFormat << " matrix format." << std::endl;
      return 1;
    }
  }

  int MatrixHandlerCuda::Matrix1Norm(matrix::Sparse* /* A */, real_type* /* norm */)
  {
    return -1;
  }

  int MatrixHandlerCuda::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    index_type error_sum = 0;
    LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;

    A_csr->allocateMatrixData("cuda");
    index_type n = A_csc->getNumRows();
    index_type m = A_csc->getNumRows();
    index_type nnz = A_csc->getNnz();
    size_t bufferSize;
    void* d_work;
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(workspaceCUDA->getCusparseHandle(),
                                                            n, 
                                                            m, 
                                                            nnz, 
                                                            A_csc->getValues("cuda"), 
                                                            A_csc->getColData("cuda"), 
                                                            A_csc->getRowData("cuda"), 
                                                            A_csr->getValues("cuda"), 
                                                            A_csr->getRowData("cuda"),
                                                            A_csr->getColData("cuda"), 
                                                            CUDA_R_64F, 
                                                            CUSPARSE_ACTION_NUMERIC,
                                                            CUSPARSE_INDEX_BASE_ZERO, 
                                                            CUSPARSE_CSR2CSC_ALG1, 
                                                            &bufferSize);
    error_sum += status;
    mem_.allocateBufferOnDevice(&d_work, bufferSize);
    status = cusparseCsr2cscEx2(workspaceCUDA->getCusparseHandle(),
                                n, 
                                m, 
                                nnz, 
                                A_csc->getValues("cuda"), 
                                A_csc->getColData("cuda"), 
                                A_csc->getRowData("cuda"), 
                                A_csr->getValues("cuda"), 
                                A_csr->getRowData("cuda"),
                                A_csr->getColData("cuda"), 
                                CUDA_R_64F,
                                CUSPARSE_ACTION_NUMERIC,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUSPARSE_CSR2CSC_ALG1,
                                d_work);
    error_sum += status;
    return error_sum;
    mem_.deleteOnDevice(d_work);
  }

} // namespace ReSolve