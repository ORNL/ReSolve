#include <algorithm>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#include "MatrixHandlerCuda.hpp"
#include <resolve/cuda/cudaKernels.h>
#include <resolve/cusolver_defs.hpp> // needed for inf nrm

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


  /**
   * @brief result := alpha * A * x + beta * result
   * 
   * @param[in]     A - matrix
   * @param[in]     vec_x - vector multiplied by A
   * @param[in,out] vec_result - resulting vector
   * @param[in]     alpha - matrix-vector multiplication factor
   * @param[in]     beta - sum into result factor
   * @return int    error code, 0 if successful
   * 
   * @pre Matrix `A` is in CSR format.
   * 
   * @note If we decide to implement this function for different matrix
   * format, the check for CSR matrix will be replaced with a switch
   * statement to select implementation for recognized input matrix
   * format.
   */
  int MatrixHandlerCuda::matvec(matrix::Sparse* A, 
                                vector_type* vec_x, 
                                vector_type* vec_result, 
                                const real_type* alpha, 
                                const real_type* beta) 
  {
    using namespace constants;

    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
           "Matrix has to be in CSR format for matrix-vector product.\n");

    int error_sum = 0;
    //result = alpha *A*x + beta * result
    cusparseStatus_t status;
    cusparseDnVecDescr_t vecx = workspace_->getVecX();
    cusparseCreateDnVec(&vecx, A->getNumRows(), vec_x->getData(memory::DEVICE), CUDA_R_64F);


    cusparseDnVecDescr_t vecAx = workspace_->getVecY();
    cusparseCreateDnVec(&vecAx, A->getNumRows(), vec_result->getData(memory::DEVICE), CUDA_R_64F);

    cusparseSpMatDescr_t matA = workspace_->getSpmvMatrixDescriptor();

    void* buffer_spmv = workspace_->getSpmvBuffer();
    cusparseHandle_t handle_cusparse = workspace_->getCusparseHandle();
    if (values_changed_) {
      status = cusparseCreateCsr(&matA, 
                                 A->getNumRows(),
                                 A->getNumColumns(),
                                 A->getNnz(),
                                 A->getRowData(memory::DEVICE),
                                 A->getColData(memory::DEVICE),
                                 A->getValues( memory::DEVICE), 
                                 CUSPARSE_INDEX_32I, 
                                 CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO,
                                 CUDA_R_64F);
      error_sum += status;
      values_changed_ = false;
    }
    if (!workspace_->matvecSetup()) {
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
      workspace_->setSpmvMatrixDescriptor(matA);
      workspace_->setSpmvBuffer(buffer_spmv);

      workspace_->matvecSetupDone();
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
      out::error() << "Matvec status: "   << status                    << ". "
                   << "Last error code: " << mem_.getLastDeviceError() << ".\n";
    vec_result->setDataUpdated(memory::DEVICE);

    cusparseDestroyDnVec(vecx);
    cusparseDestroyDnVec(vecAx);
    return error_sum;
  }

  /**
   * @brief Matrix infinity norm
   * 
   * @param[in]  A - matrix
   * @param[out] norm - matrix norm
   * @return int error code, 0 if successful
   * 
   * @pre Matrix `A` is in CSR format.
   * 
   * @note If we decide to implement this function for different matrix
   * format, the check for CSR matrix will be replaced with a switch
   * statement to select implementation for recognized input matrix
   * format.
   */
  int MatrixHandlerCuda::matrixInfNorm(matrix::Sparse* A, real_type* norm)
  {
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
           "Matrix has to be in CSR format for matrix-vector product.\n");

    if (workspace_->getNormBufferState() == false) { // not allocated  
      real_type* buffer;
      mem_.allocateArrayOnDevice(&buffer, 1024);
      workspace_->setNormBuffer(buffer);
      workspace_->setNormBufferState(true);
    }

    real_type* d_r = workspace_->getDr();
    if (workspace_->getDrSize() != A->getNumRows()) {
      if (d_r != nullptr) {
        mem_.deleteOnDevice(d_r);
      }
      mem_.allocateArrayOnDevice(&d_r, A->getNumRows());
      workspace_->setDrSize(A->getNumRows());
      workspace_->setDr(d_r);
    }

    matrix_row_sums(A->getNumRows(),
                    A->getNnz(),
                    A->getRowData(memory::DEVICE),
                    A->getValues(memory::DEVICE),
                    d_r);

    int status = cusolverSpDnrm_inf(workspace_->getCusolverSpHandle(),
                                   A->getNumRows(),
                                   d_r,
                                   norm,
                                   workspace_->getNormBuffer()  /* at least 8192 bytes */);

    if (status != 0) {
      io::Logger::warning() << "Vector inf nrm returned "<<status<<"\n"; 
    }
    return status;
  }

  int MatrixHandlerCuda::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    index_type error_sum = 0;

    A_csr->allocateMatrixData(memory::DEVICE);
    index_type n = A_csc->getNumRows();
    index_type m = A_csc->getNumRows();
    index_type nnz = A_csc->getNnz();
    size_t bufferSize;
    void* d_work;
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(workspace_->getCusparseHandle(),
                                                            n, 
                                                            m, 
                                                            nnz, 
                                                            A_csc->getValues( memory::DEVICE), 
                                                            A_csc->getColData(memory::DEVICE), 
                                                            A_csc->getRowData(memory::DEVICE), 
                                                            A_csr->getValues( memory::DEVICE), 
                                                            A_csr->getRowData(memory::DEVICE),
                                                            A_csr->getColData(memory::DEVICE), 
                                                            CUDA_R_64F, 
                                                            CUSPARSE_ACTION_NUMERIC,
                                                            CUSPARSE_INDEX_BASE_ZERO, 
                                                            CUSPARSE_CSR2CSC_ALG1, 
                                                            &bufferSize);
    error_sum += status;
    mem_.allocateBufferOnDevice(&d_work, bufferSize);
    status = cusparseCsr2cscEx2(workspace_->getCusparseHandle(),
                                n, 
                                m, 
                                nnz, 
                                A_csc->getValues( memory::DEVICE), 
                                A_csc->getColData(memory::DEVICE), 
                                A_csc->getRowData(memory::DEVICE), 
                                A_csr->getValues( memory::DEVICE), 
                                A_csr->getRowData(memory::DEVICE),
                                A_csr->getColData(memory::DEVICE), 
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
