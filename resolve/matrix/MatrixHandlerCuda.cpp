#include "MatrixHandlerCuda.hpp"

#include <algorithm>
#include <numeric>

#include <resolve/cuda/cudaKernels.h>
#include <resolve/cuda/cudaVectorKernels.h>

#include <resolve/cusolver_defs.hpp> // needed for inf nrm
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>

namespace ReSolve
{
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  /**
   * @brief Empty constructor for MatrixHandlerCuda object
   */
  MatrixHandlerCuda::~MatrixHandlerCuda()
  {
  }

  /**
   * @brief Constructor for MatrixHandlerCuda object
   *
   * @param[in] new_workspace - pointer to the workspace object
   */
  MatrixHandlerCuda::MatrixHandlerCuda(LinAlgWorkspaceCUDA* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /**
   * @brief Set values changed flag
   *
   * @param[in] values_changed - flag indicating if values have changed
   */
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
  int MatrixHandlerCuda::matvec(matrix::Sparse*  A,
                                vector_type*     vec_x,
                                vector_type*     vec_result,
                                const real_type* alpha,
                                const real_type* beta)
  {
    using namespace constants;

    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for matrix-vector product.\n");

    int error_sum = 0;
    // result = alpha *A*x + beta * result
    cusparseStatus_t     status;
    cusparseDnVecDescr_t vecx = workspace_->getVecX();
    cusparseCreateDnVec(&vecx, A->getNumRows(), vec_x->getData(memory::DEVICE), CUDA_R_64F);

    cusparseDnVecDescr_t vecAx = workspace_->getVecY();
    cusparseCreateDnVec(&vecAx, A->getNumRows(), vec_result->getData(memory::DEVICE), CUDA_R_64F);

    cusparseSpMatDescr_t matA = workspace_->getSpmvMatrixDescriptor();

    void*            buffer_spmv     = workspace_->getSpmvBuffer();
    cusparseHandle_t handle_cusparse = workspace_->getCusparseHandle();
    if (values_changed_)
    {
      status = cusparseCreateCsr(&matA,
                                 A->getNumRows(),
                                 A->getNumColumns(),
                                 A->getNnz(),
                                 A->getRowData(memory::DEVICE),
                                 A->getColData(memory::DEVICE),
                                 A->getValues(memory::DEVICE),
                                 CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO,
                                 CUDA_R_64F);
      error_sum += status;
      values_changed_ = false;
    }
    if (!workspace_->matvecSetup())
    {
      // setup first, allocate, etc.
      size_t bufferSize = 0;

      status = cusparseSpMV_bufferSize(handle_cusparse,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &MINUS_ONE,
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
      out::error() << "Matvec status: " << status << ". "
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
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for matrix-vector product.\n");

    if (workspace_->getNormBufferState() == false)
    { // not allocated
      real_type* buffer;
      mem_.allocateArrayOnDevice(&buffer, 1024);
      workspace_->setNormBuffer(buffer);
      workspace_->setNormBufferState(true);
    }

    real_type* d_r = workspace_->getDr();
    if (workspace_->getDrSize() != A->getNumRows())
    {
      if (d_r != nullptr)
      {
        mem_.deleteOnDevice(d_r);
      }
      mem_.allocateArrayOnDevice(&d_r, A->getNumRows());
      workspace_->setDrSize(A->getNumRows());
      workspace_->setDr(d_r);
    }

    cuda::matrix_row_sums(A->getNumRows(),
                          A->getNnz(),
                          A->getRowData(memory::DEVICE),
                          A->getValues(memory::DEVICE),
                          d_r);

    int status = cusolverSpDnrminf(workspace_->getCusolverSpHandle(),
                                   A->getNumRows(),
                                   d_r,
                                   norm,
                                   workspace_->getNormBuffer() /* at least 8192 bytes */);

    if (status != 0)
    {
      io::Logger::warning() << "Vector inf nrm returned " << status << "\n";
    }
    return status;
  }

  /**
   * @brief convert a CSC matrix to a CSR matrix in CUDA
   *
   * @param[in]  A_csc - input CSC matrix
   * @param[out] A_csr - output CSR matrix
   * @return int error_sum, 0 if successful
   */
  int MatrixHandlerCuda::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    index_type error_sum = 0;

    A_csr->allocateMatrixData(memory::DEVICE);
    index_type m   = A_csc->getNumColumns();
    index_type n   = A_csc->getNumRows();
    index_type nnz = A_csc->getNnz();

    // check dimensions of A_csc and A_csr
    assert(A_csc->getNumRows() == A_csr->getNumRows() && "Number of rows in A_csc must be equal to number of rows in A_csr");
    assert(A_csc->getNumColumns() == A_csr->getNumColumns() && "Number of columns in A_csc must be equal to number of columns in A_csr");

    size_t           bufferSize;
    void*            d_work;
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(workspace_->getCusparseHandle(),
                                                            m,
                                                            n,
                                                            nnz,
                                                            A_csc->getValues(memory::DEVICE),
                                                            A_csc->getColData(memory::DEVICE),
                                                            A_csc->getRowData(memory::DEVICE),
                                                            A_csr->getValues(memory::DEVICE),
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
                                m,
                                n,
                                nnz,
                                A_csc->getValues(memory::DEVICE),
                                A_csc->getColData(memory::DEVICE),
                                A_csc->getRowData(memory::DEVICE),
                                A_csr->getValues(memory::DEVICE),
                                A_csr->getRowData(memory::DEVICE),
                                A_csr->getColData(memory::DEVICE),
                                CUDA_R_64F,
                                CUSPARSE_ACTION_NUMERIC,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUSPARSE_CSR2CSC_ALG1,
                                d_work);
    error_sum += status;
    if (status)
    {
      out::error() << "CSC2CSR status: " << status << ". "
                   << "Last error code: " << mem_.getLastDeviceError() << ".\n";
    }
    mem_.deleteOnDevice(d_work);

    // Values on the device are updated now -- mark them as such!
    A_csr->setUpdated(memory::DEVICE);

    return error_sum;
  }

  /**
   * @brief Transpose a sparse CSR matrix.
   *
   * Transpose a sparse CSR matrix A. Only allocates At if not already allocated.
   *
   * @param[in, out]  A - Sparse matrix
   * @param[out] At - Transposed matrix
   *
   * @return int error_sum, 0 if successful
   */
  int MatrixHandlerCuda::transpose(matrix::Csr* A, matrix::Csr* At)
  {
    index_type       error_sum = 0;
    index_type       m         = A->getNumRows();
    index_type       n         = A->getNumColumns();
    index_type       nnz       = A->getNnz();
    cusparseStatus_t status;
    bool             allocated = workspace_->isTransposeBufferAllocated();
    // check dimensions of A and At
    if (!allocated)
    {
      // allocate transpose buffer
      size_t bufferSize;
      status = cusparseCsr2cscEx2_bufferSize(workspace_->getCusparseHandle(),
                                             m,
                                             n,
                                             nnz,
                                             A->getValues(memory::DEVICE),
                                             A->getRowData(memory::DEVICE),
                                             A->getColData(memory::DEVICE),
                                             At->getValues(memory::DEVICE),
                                             At->getRowData(memory::DEVICE),
                                             At->getColData(memory::DEVICE),
                                             CUDA_R_64F,
                                             CUSPARSE_ACTION_NUMERIC,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUSPARSE_CSR2CSC_ALG1,
                                             &bufferSize);
      error_sum += status;
      workspace_->setTransposeBufferWorkspace(bufferSize);
    }
    status = cusparseCsr2cscEx2(workspace_->getCusparseHandle(),
                                m,
                                n,
                                nnz,
                                A->getValues(memory::DEVICE),
                                A->getRowData(memory::DEVICE),
                                A->getColData(memory::DEVICE),
                                At->getValues(memory::DEVICE),
                                At->getRowData(memory::DEVICE),
                                At->getColData(memory::DEVICE),
                                CUDA_R_64F,
                                CUSPARSE_ACTION_NUMERIC,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUSPARSE_CSR2CSC_ALG1,
                                workspace_->getTransposeBufferWorkspace());
    error_sum += status;
    // Values on the device are updated now -- mark them as such!
    At->setUpdated(memory::DEVICE);

    return error_sum;
  }

  /**
   * @brief Left diagonal scaling of a sparse CSR matrix in CUDA
   *
   * @param[in]  diag - vector representing the diagonal matrix
   * @param[in, out]  A - Sparse CSR matrix
   *
   * @pre The diagonal vector must be of the same size as the number of rows in the matrix.
   * @pre A is unscaled and allocated
   * @post A is scaled
   * @invariant diag
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCuda::leftScale(vector_type* diag, matrix::Csr* A)
  {
    real_type*  diag_data = diag->getData(memory::DEVICE);
    index_type* a_row_ptr = A->getRowData(memory::DEVICE);
    real_type*  a_vals    = A->getValues(memory::DEVICE);
    index_type  n         = A->getNumRows();
    // check values in A and diag
    cuda::leftScale(n, a_row_ptr, a_vals, diag_data);
    A->setUpdated(memory::DEVICE);
    return 0;
  }

  /**
   * @brief Right diagonal scaling of a sparse CSR matrix in CUDA
   *
   * @param[in]  A - Sparse CSR matrix
   * @param[in]  diag - vector representing the diagonal matrix
   *
   * @pre The diagonal vector must be of the same size as the number of columns in the matrix.
   * @pre A is unscaled
   * @post A is scaled
   * @invariant diag
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCuda::rightScale(matrix::Csr* A, vector_type* diag)
  {
    real_type*  diag_data = diag->getData(memory::DEVICE);
    index_type* a_row_ptr = A->getRowData(memory::DEVICE);
    index_type* a_col_idx = A->getColData(memory::DEVICE);
    real_type*  a_vals    = A->getValues(memory::DEVICE);
    index_type  n         = A->getNumRows();
    cuda::rightScale(n, a_row_ptr, a_col_idx, a_vals, diag_data);
    A->setUpdated(memory::DEVICE);
    return 0;
  }

  /**
   * @brief Add a constant to all nonzero values in the matrix
   *
   * @param[in, out] A - matrix
   * @param[in] alpha - constant to be added
   *
   * @return int error code, 0 if successful
   */
  int MatrixHandlerCuda::addConst(matrix::Sparse* A, real_type alpha)
  {
    real_type* values = A->getValues(memory::DEVICE);
    index_type nnz    = A->getNnz();
    cuda::addConst(nnz, alpha, values);
    return 0;
  }

  void MatrixHandlerCuda::allocateForSum(matrix::Csr* A, real_type alpha, matrix::Csr* B, real_type beta, matrix::Csr* C)
  {
    auto handle = workspace_->getCusparseHandle();
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    workspace_->setCusparseHandle(handle);

    cusparseMatDescr_t descr_a = workspace_->getScaleAddMatrixDescriptor();
    cusparseCreateMatDescr(&descr_a);
    cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
    workspace_->setScaleAddMatrixDescriptor(descr_a);

    auto a_v = A->getValues(memory::DEVICE);
    auto a_i = A->getRowData(memory::DEVICE);
    auto a_j = A->getColData(memory::DEVICE);

    auto b_v = B->getValues(memory::DEVICE);
    auto b_i = B->getRowData(memory::DEVICE);
    auto b_j = B->getColData(memory::DEVICE);

    index_type m = A->getNumRows();
    assert(m == B->getNumRows());
    index_type n = A->getNumColumns();
    assert(n == B->getNumColumns());

    assert(m == n);

    index_type nnz_a = A->getNnz();
    index_type nnz_b = B->getNnz();

    real_type*  c_v = nullptr;
    index_type* c_i = nullptr;
    index_type* c_j = nullptr;

    size_t buffer_byte_size_add;
    // calculates sum buffer
    cusparseStatus_t info = cusparseDcsrgeam2_bufferSizeExt(handle, m, n, &alpha, descr_a, nnz_a, a_v, a_i, a_j, &beta, descr_a, nnz_b, b_v, b_i, b_j, descr_a, c_v, c_i, c_j, &buffer_byte_size_add);
    assert(info == CUSPARSE_STATUS_SUCCESS);

    auto buffer = new ScaleAddBufferCUDA(n + 1, buffer_byte_size_add);
    workspace_->setScaleAddBBuffer(buffer);

    index_type nnz_total;
    // determines sum row offsets and total number of nonzeros
    info = cusparseXcsrgeam2Nnz(handle, m, n, descr_a, nnz_a, a_i, a_j, descr_a, nnz_b, b_i, b_j, descr_a, buffer->getRowData(), &nnz_total, buffer->getBuffer());
    assert(info == CUSPARSE_STATUS_SUCCESS);

    C->setNnz(nnz_total);
    C->allocateMatrixData(memory::DEVICE);
    mem_.copyArrayDeviceToDevice(C->getRowData(memory::DEVICE), buffer->getRowData(), n + 1);
  }

  void MatrixHandlerCuda::computeSum(matrix::Csr* A, real_type alpha, matrix::Csr* B, real_type beta, matrix::Csr* C)
  {
    auto a_v = A->getValues(memory::DEVICE);
    auto a_i = A->getRowData(memory::DEVICE);
    auto a_j = A->getColData(memory::DEVICE);

    auto b_v = B->getValues(memory::DEVICE);
    auto b_i = B->getRowData(memory::DEVICE);
    auto b_j = B->getColData(memory::DEVICE);

    auto c_v = C->getValues(memory::DEVICE);
    auto c_i = C->getRowData(memory::DEVICE);
    auto c_j = C->getColData(memory::DEVICE);

    auto handle = workspace_->getCusparseHandle();

    index_type m = A->getNumRows();
    assert(m == B->getNumRows());
    assert(m == C->getNumRows());

    index_type n = A->getNumColumns();
    assert(n == B->getNumColumns());
    assert(n == C->getNumColumns());

    assert(m == n);

    index_type nnz_a = A->getNnz();
    index_type nnz_b = B->getNnz();

    ScaleAddBufferCUDA* buffer = workspace_->getScaleAddBBuffer();
    mem_.copyArrayDeviceToDevice(c_i, buffer->getRowData(), n + 1);
    cusparseMatDescr_t descr_a = workspace_->getScaleAddMatrixDescriptor();
    cusparseStatus_t   info    = cusparseDcsrgeam2(handle, m, n, &alpha, descr_a, nnz_a, a_v, a_i, a_j, &beta, descr_a, nnz_b, b_v, b_i, b_j, descr_a, c_v, c_i, c_j, buffer->getBuffer());
    assert(info == CUSPARSE_STATUS_SUCCESS);
    C->setUpdated(memory::DEVICE);
  }

  /**
   * @brief Update matrix with a new number of nonzero elements
   *
   * @param[in,out] A - Sparse matrix
   * @param[in] row_data - pointer to row data (array of integers)
   * @param[in] col_data - pointer to column data (array of integers)
   * @param[in] val_data - pointer to value data (array of real numbers)
   * @param[in] nnz - number of non-zer
   * @return 0 if successful, 1 otherwise
   */
  static int updateMatrix(matrix::Sparse* A, index_type* rowData, index_type* columnData, real_type* valData, index_type nnz)
  {
    if (A->destroyMatrixData(memory::DEVICE) != 0)
    {
      return 1;
    }
    if (A->destroyMatrixData(memory::HOST) != 0)
    {
      return 1;
    }
    return A->copyDataFrom(rowData, columnData, valData, nnz, memory::DEVICE, memory::DEVICE);
  }

  /**
   * @brief Add a constant to the nonzero values of a csr matrix,
   *        then add the identity matrix.
   *
   * @param[in,out] A - Sparse CSR matrix
   * @param[in] alpha - constant to the added
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCuda::scaleAddI(matrix::Csr* A, real_type alpha)
  {
    index_type n = A->getNumRows();

    std::vector<index_type> I_i(n + 1);
    std::iota(I_i.begin(), I_i.end(), 0);

    std::vector<index_type> I_j(n);
    std::iota(I_j.begin(), I_j.end(), 0);

    std::vector<real_type>  I_v(n, 1.0);

    matrix::Csr I(A->getNumRows(), A->getNumColumns(), n);
    I.copyDataFrom(I_i.data(), I_j.data(), I_v.data(), n, memory::HOST, memory::DEVICE);

    return scaleAddB(A, alpha, &I);
  }

  /**
   * @brief Multiply csr matrix by a constant and add B.
   *
   * @param[in,out] A - Sparse CSR matrix
   * @param[in] alpha - constant to the added
   * @param[in] B - Sparse CSR matrix
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCuda::scaleAddB(matrix::Csr* A, real_type alpha, matrix::Csr* B)
  {
    matrix::Csr C(A->getNumRows(), A->getNumColumns(), A->getNnz());
    allocateForSum(A, alpha, B, 1., &C);

    computeSum(A, alpha, B, 1., &C);

    updateMatrix(A, C.getRowData(memory::DEVICE), C.getColData(memory::DEVICE), C.getValues(memory::DEVICE), C.getNnz());
    return 0;
  }

} // namespace ReSolve
