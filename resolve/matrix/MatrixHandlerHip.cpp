#include "MatrixHandlerHip.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

#include <resolve/hip/hipKernels.h>
#include <resolve/hip/hipVectorKernels.h>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>

namespace ReSolve
{
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  /**
   * @brief Empty constructor for MatrixHandlerHip object
   */
  MatrixHandlerHip::~MatrixHandlerHip()
  {
  }

  /**
   * @brief Constructor for MatrixHandlerHip object
   *
   * @param[in] new_workspace - pointer to the workspace object
   */
  MatrixHandlerHip::MatrixHandlerHip(LinAlgWorkspaceHIP* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /**
   * @brief Set values changed flag
   *
   * @param[in] values_changed - flag indicating if values have changed
   */
  void MatrixHandlerHip::setValuesChanged(bool values_changed)
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
  int MatrixHandlerHip::matvec(matrix::Sparse*  A,
                               vector_type*     vec_x,
                               vector_type*     vec_result,
                               const real_type* alpha,
                               const real_type* beta)
  {
    using namespace constants;

    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for matrix-vector product.\n");

    int error_sum = 0;
    // result = alpha *A*x + beta * result
    rocsparse_status status;

    rocsparse_handle handle_rocsparse = workspace_->getRocsparseHandle();

    rocsparse_mat_info  infoA  = workspace_->getSpmvMatrixInfo();
    rocsparse_mat_descr descrA = workspace_->getSpmvMatrixDescriptor();

    if (!workspace_->matvecSetup())
    {
      // setup first, allocate, etc.
      rocsparse_create_mat_descr(&(descrA));
      rocsparse_set_mat_index_base(descrA, rocsparse_index_base_zero);
      rocsparse_set_mat_type(descrA, rocsparse_matrix_type_general);

      rocsparse_create_mat_info(&infoA);

      status = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                         rocsparse_operation_none,
                                         A->getNumRows(),
                                         A->getNumColumns(),
                                         A->getNnz(),
                                         descrA,
                                         A->getValues(memory::DEVICE),
                                         A->getRowData(memory::DEVICE),
                                         A->getColData(memory::DEVICE),
                                         infoA);
      error_sum += status;
      mem_.deviceSynchronize();

      workspace_->setSpmvMatrixDescriptor(descrA);
      workspace_->setSpmvMatrixInfo(infoA);
      workspace_->matvecSetupDone();
    }

    status = rocsparse_dcsrmv(handle_rocsparse,
                              rocsparse_operation_none,
                              A->getNumRows(),
                              A->getNumColumns(),
                              A->getNnz(),
                              alpha,
                              descrA,
                              A->getValues(memory::DEVICE),
                              A->getRowData(memory::DEVICE),
                              A->getColData(memory::DEVICE),
                              infoA,
                              vec_x->getData(memory::DEVICE),
                              beta,
                              vec_result->getData(memory::DEVICE));

    error_sum += status;
    mem_.deviceSynchronize();
    if (status)
    {
      out::error() << "Matvec status: " << status << ". "
                   << "Last error code: " << mem_.getLastDeviceError() << ".\n";
    }
    vec_result->setDataUpdated(memory::DEVICE);

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
  int MatrixHandlerHip::matrixInfNorm(matrix::Sparse* A, real_type* norm)
  {
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for matrix-vector product.\n");

    real_type* d_r      = workspace_->getDr();
    index_type d_r_size = workspace_->getDrSize();

    if (d_r_size != A->getNumRows())
    {
      if (d_r_size != 0)
      {
        mem_.deleteOnDevice(d_r);
      }
      mem_.allocateArrayOnDevice(&d_r, A->getNumRows());
      workspace_->setDrSize(A->getNumRows());
      workspace_->setDr(d_r);
    }

    if (workspace_->getNormBufferState() == false)
    { // not allocated
      real_type* buffer;
      mem_.allocateArrayOnDevice(&buffer, 1024);
      workspace_->setNormBuffer(buffer);
      workspace_->setNormBufferState(true);
    }

    mem_.deviceSynchronize();
    hip::matrix_row_sums(A->getNumRows(),
                         A->getNnz(),
                         A->getRowData(memory::DEVICE),
                         A->getValues(memory::DEVICE),
                         d_r);
    mem_.deviceSynchronize();

    hip::vector_inf_norm(A->getNumRows(),
                         d_r,
                         workspace_->getNormBuffer(),
                         norm);
    return 0;
  }

  /**
   * @brief convert a CSC matrix to a CSR matrix in HIP
   *
   * @param[in]  A_csc - input CSC matrix
   * @param[out] A_csr - output CSR matrix
   * @return int error_sum, 0 if successful
   */
  int MatrixHandlerHip::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    index_type error_sum = 0;

    rocsparse_status status;

    A_csr->allocateMatrixData(memory::DEVICE);
    index_type m   = A_csc->getNumColumns();
    index_type n   = A_csc->getNumRows();
    index_type nnz = A_csc->getNnz();
    size_t     bufferSize;
    void*      d_work;

    status = rocsparse_csr2csc_buffer_size(workspace_->getRocsparseHandle(),
                                           m,
                                           n,
                                           nnz,
                                           A_csc->getColData(memory::DEVICE),
                                           A_csc->getRowData(memory::DEVICE),
                                           rocsparse_action_numeric,
                                           &bufferSize);

    error_sum += status;
    mem_.allocateBufferOnDevice(&d_work, bufferSize);

    status = rocsparse_dcsr2csc(workspace_->getRocsparseHandle(),
                                m,
                                n,
                                nnz,
                                A_csc->getValues(memory::DEVICE),
                                A_csc->getColData(memory::DEVICE),
                                A_csc->getRowData(memory::DEVICE),
                                A_csr->getValues(memory::DEVICE),
                                A_csr->getColData(memory::DEVICE),
                                A_csr->getRowData(memory::DEVICE),
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                d_work);
    error_sum += status;
    mem_.deleteOnDevice(d_work);

    // Values on the device are updated now -- mark them as such!
    A_csr->setUpdated(memory::DEVICE);
    mem_.deviceSynchronize();

    return error_sum;
  }

  /**
   * @brief Transpose a sparse CSR matrix (HIP backend).
   *
   * Transpose a sparse CSR matrix A. Only allocates At if not already allocated.
   *
   * @param[in, out]  A - Sparse matrix
   * @param[out]      At - Transposed matrix
   *
   * @return int error_sum, 0 if successful
   *
   * @warning This method works only for `real_type == double`.
   */
  int MatrixHandlerHip::transpose(matrix::Csr* A, matrix::Csr* At)
  {
    index_type       error_sum = 0;
    index_type       m         = A->getNumRows();
    index_type       n         = A->getNumColumns();
    index_type       nnz       = A->getNnz();
    rocsparse_status status;
    bool             allocated = workspace_->isTransposeBufferAllocated();
    if (!allocated)
    {
      // allocate transpose buffer
      size_t bufferSize;
      status = rocsparse_csr2csc_buffer_size(workspace_->getRocsparseHandle(),
                                             m,
                                             n,
                                             nnz,
                                             A->getRowData(memory::DEVICE),
                                             A->getColData(memory::DEVICE),
                                             rocsparse_action_numeric,
                                             &bufferSize);
      error_sum += status;
      workspace_->setTransposeBufferWorkspace(bufferSize);
    }
    status = rocsparse_dcsr2csc(workspace_->getRocsparseHandle(),
                                m,
                                n,
                                nnz,
                                A->getValues(memory::DEVICE),
                                A->getRowData(memory::DEVICE),
                                A->getColData(memory::DEVICE),
                                At->getValues(memory::DEVICE),
                                At->getColData(memory::DEVICE),
                                At->getRowData(memory::DEVICE),
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                workspace_->getTransposeBufferWorkspace());
    error_sum += status;
    // Values on the device are updated now -- mark them as such!
    At->setUpdated(memory::DEVICE);
    mem_.deviceSynchronize();

    return error_sum;
  }

  /**
   * @brief Add a constant to all nonzero values in the matrix
   *
   * @param[in, out] A - matrix
   * @param[in] alpha - constant to be added
   *
   * @return int error code, 0 if successful
   */
  int MatrixHandlerHip::addConst(matrix::Sparse* A, real_type alpha)
  {
    real_type* values = A->getValues(memory::DEVICE);
    index_type nnz    = A->getNnz();
    hip::addConst(nnz, alpha, values);
    mem_.deviceSynchronize();
    return 0;
  }

  /**
   * @brief Left diagonal scaling of a sparse CSR matrix in HIP
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
  int MatrixHandlerHip::leftScale(vector_type* diag, matrix::Csr* A)
  {
    real_type*  diag_data = diag->getData(memory::DEVICE);
    index_type* a_row_ptr = A->getRowData(memory::DEVICE);
    real_type*  a_vals    = A->getValues(memory::DEVICE);
    index_type  n         = A->getNumRows();
    // check values in A and diag
    hip::leftScale(n, a_row_ptr, a_vals, diag_data);
    A->setUpdated(memory::DEVICE);
    mem_.deviceSynchronize();
    return 0;
  }

  /**
   * @brief Right diagonal scaling of a sparse CSR matrix in HIP
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
  int MatrixHandlerHip::rightScale(matrix::Csr* A, vector_type* diag)
  {
    real_type*  diag_data = diag->getData(memory::DEVICE);
    index_type* a_row_ptr = A->getRowData(memory::DEVICE);
    index_type* a_col_idx = A->getColData(memory::DEVICE);
    real_type*  a_vals    = A->getValues(memory::DEVICE);
    index_type  n         = A->getNumRows();
    hip::rightScale(n, a_row_ptr, a_col_idx, a_vals, diag_data);
    A->setUpdated(memory::DEVICE);
    mem_.deviceSynchronize();
    return 0;
  }

  /**
   * @brief Calculate buffer size and sparsity pattern of alpha*A + beta*B
   *
   * @param[in] A - CSR matrix
   * @param[in] alpha - constant to be added to A
   * @param[in] B - CSR matrix
   * @param[in] beta - constant to be added to B
   * @param[in, out] pattern - sparsity pattern and buffer
   *
   * @return int error code, 0 if successful
   */
  int MatrixHandlerHip::allocateForSum(matrix::Csr* A, real_type alpha, matrix::Csr* B, real_type beta, ScaleAddBufferHIP** pattern)
  {
    auto             handle  = workspace_->getRocsparseHandle();
    rocsparse_status roc_info = rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
    int              info    = (roc_info != rocsparse_status_success);
    workspace_->setRocsparseHandle(handle);

    rocsparse_mat_descr descr_a = workspace_->getScaleAddMatrixDescriptor();
    rocsparse_create_mat_descr(&descr_a);
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

    size_t buffer_byte_size_add{0};
    *pattern = new ScaleAddBufferHIP(n + 1, buffer_byte_size_add);

    index_type nnz_total;
    // determines sum row offsets and total number of nonzeros
    roc_info = rocsparse_bsrgeam_nnzb(handle, rocsparse_direction_row, m, n, 1, descr_a, nnz_a, a_i, a_j, descr_a, nnz_b, b_i, b_j, descr_a, (*pattern)->getRowData(), &nnz_total);
    info    = info || (roc_info != rocsparse_status_success);
    (*pattern)->setNnz(nnz_total);
    return 0;
  }


  /**
   * @brief Given sparsity pattern, calculate alpha*A + beta*B
   *
   * @param[in] A - CSR matrix
   * @param[in] alpha - constant to be added to A
   * @param[in] B - CSR matrix
   * @param[in] beta - constant to be added to B
   * @param[in, out] pattern - sparsity pattern and buffer
   *
   * @return int error code, 0 if successful
   */
  int MatrixHandlerHip::computeSum(matrix::Csr* A, real_type alpha, matrix::Csr* B, real_type beta, matrix::Csr* C, ScaleAddBufferHIP* pattern)
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

    auto handle = workspace_->getRocsparseHandle();

    index_type m = A->getNumRows();
    assert(m == B->getNumRows());
    assert(m == C->getNumRows());

    index_type n = A->getNumColumns();
    assert(n == B->getNumColumns());
    assert(n == C->getNumColumns());

    assert(m == n);

    index_type nnz_a = A->getNnz();
    index_type nnz_b = B->getNnz();

    int                info    = mem_.copyArrayDeviceToDevice(c_i, pattern->getRowData(), n + 1);
    rocsparse_mat_descr descr_a = workspace_->getScaleAddMatrixDescriptor();
    rocsparse_status roc_info = rocsparse_dbsrgeam(handle, rocsparse_direction_row, m, n, 1, &alpha, descr_a, nnz_a, a_v, a_i, a_j, &beta, descr_a, nnz_b, b_v, b_i, b_j, descr_a, c_v, c_i, c_j);
    info                       = info || (roc_info != rocsparse_status_success);
    info                       = info || C->setUpdated(memory::DEVICE);
    return info;
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
  int MatrixHandlerHip::scaleAddI(matrix::Csr* A, real_type alpha)
  {
    index_type n = A->getNumRows();

    std::vector<index_type> I_i(n + 1);
    std::iota(I_i.begin(), I_i.end(), 0);

    std::vector<index_type> I_j(n);
    std::iota(I_j.begin(), I_j.end(), 0);

    std::vector<real_type> I_v(n, 1.0);

    matrix::Csr I(A->getNumRows(), A->getNumColumns(), n);
    int         info = I.copyDataFrom(I_i.data(), I_j.data(), I_v.data(), n, memory::HOST, memory::DEVICE);

    // Reuse sparsity pattern if it is available
    if (workspace_->scaleAddISetup())
    {
      ScaleAddBufferHIP* pattern = workspace_->getScaleAddIBuffer();
      matrix::Csr C(A->getNumRows(), A->getNumColumns(), pattern->getNnz());
      info = info || C.allocateMatrixData(memory::DEVICE);
      info = info || computeSum(A, alpha, &I, 1., &C, pattern);
      info = info || updateMatrix(A, C.getRowData(memory::DEVICE), C.getColData(memory::DEVICE), C.getValues(memory::DEVICE), C.getNnz());
    }
    else
    {
      ScaleAddBufferHIP* pattern = nullptr;
      matrix::Csr C(A->getNumRows(), A->getNumColumns(), A->getNnz());
      info = info || allocateForSum(A, alpha, &I, 1., &pattern);
      workspace_->setScaleAddIBuffer(pattern);
      workspace_->scaleAddISetupDone();
      C.setNnz(pattern->getNnz());
      C.allocateMatrixData(memory::DEVICE);
      info = info || computeSum(A, alpha, &I, 1., &C, pattern);
      info = info || updateMatrix(A, C.getRowData(memory::DEVICE), C.getColData(memory::DEVICE), C.getValues(memory::DEVICE), C.getNnz());
    }
    return info;
  }

  /**
   * @brief Multiply csr matrix by a constant and add B.
   *
   * @param[in,out] A - Sparse CSR matrix
   * @param[in] alpha - constant to the added
   * @param[in] B - Sparse CSR matrix
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerHip::scaleAddB(matrix::Csr* A, real_type alpha, matrix::Csr* B)
  {
    int info = 0;
    // Reuse sparsity pattern if it is available
    if (workspace_->scaleAddBSetup())
    {
      ScaleAddBufferHIP* pattern = workspace_->getScaleAddBBuffer();
      matrix::Csr C(A->getNumRows(), A->getNumColumns(), pattern->getNnz());
      info = info || C.allocateMatrixData(memory::DEVICE);
      info = info || computeSum(A, alpha, B, 1., &C, pattern);
      info = info || updateMatrix(A, C.getRowData(memory::DEVICE), C.getColData(memory::DEVICE), C.getValues(memory::DEVICE), C.getNnz());
    }
    else
    {
      ScaleAddBufferHIP* pattern = nullptr;
      matrix::Csr C(A->getNumRows(), A->getNumColumns(), A->getNnz());
      info = info || allocateForSum(A, alpha, B, 1., &pattern);
      workspace_->setScaleAddBBuffer(pattern);
      workspace_->scaleAddBSetupDone();
      C.setNnz(pattern->getNnz());
      C.allocateMatrixData(memory::DEVICE);
      info = info || computeSum(A, alpha, B, 1., &C, pattern);
      info = info || updateMatrix(A, C.getRowData(memory::DEVICE), C.getColData(memory::DEVICE), C.getValues(memory::DEVICE), C.getNnz());
    }
    return info;
  }

} // namespace ReSolve
