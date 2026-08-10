#include "MatrixHandlerHip.hpp"

#include <algorithm>
#include <cassert>

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

    // The workspace caches one backend SpMV setup and temporary buffer between
    // matvec calls. SCCG can call matvec with different matrices, such as JC and
    // JC^T, so the cached setup may no longer match the current matrix structure.
    // Track the matrix pointer and dimensions/nnz so stale setup data is reset
    // before running SpMV with a different matrix.
    bool matrix_changed =
        (matrix_for_matvec_ != A) || (matvec_num_rows_ != A->getNumRows()) || (matvec_num_cols_ != A->getNumColumns()) || (matvec_nnz_ != A->getNnz());
    if (matrix_changed || values_changed_)
    {
      workspace_->resetMatvecSetup();
    }

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

      matrix_for_matvec_ = A;
      matvec_num_rows_   = A->getNumRows();
      matvec_num_cols_   = A->getNumColumns();
      matvec_nnz_        = A->getNnz();
      values_changed_    = false;
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

  /** // ... and fix comments
   * @brief result := alpha * A * x + beta * result
   *
   * @param[in]     A - matrix
   * @param[in]     vec_x - multivector multiplied by A
   * @param[in,out] vec_result - resulting multivector
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
  int MatrixHandlerHip::matMultivec(matrix::Sparse*  A,
                               vector_type*     vec_x,
                               vector_type*     vec_result,
                               const real_type* alpha,
                               const real_type* beta)
  {
  //   using namespace constants;

  //   assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for matrix-vector product.\n");

  //   int error_sum = 0;
  //   // result = alpha *A*x + beta * result
  //   rocsparse_status     status;
  //   rocsparse_dnmat_descr mat_X = workspace_->getMatX();

  //   // In SpMV, A is m x n and the operation is y = A*x so
  //   // x must have length n, the number of columns of A and
  //   // y must have length m, the number of rows of A.
  //   // This matters for non-square matrices used in SCCG.
  //   rocsparse_create_dnmat_descr(&mat_X, vec_x->getSize(), vec_x->getNumVectors(), vec_x->getSize(), vec_x->getData(memory::DEVICE), rocsparse_datatype_f64_r, rocsparse_order_column);

  //   rocsparse_dnmat_descr mat_AX = workspace_->getMatY();
  //   rocsparse_create_dnmat_descr(&mat_AX, vec_result->getSize(), vec_result->getNumVectors(), vec_result->getSize(), vec_result->getData(memory::DEVICE), rocsparse_datatype_f64_r, rocsparse_order_column);

  //   rocsparse_handle handle_cusparse = workspace_->getCusparseHandle();

  //   // The workspace caches one backend SpMV setup and temporary buffer between
  //   // matvec calls. SCCG can call matvec with different matrices, such as JC and
  //   // JC^T, so the cached setup may no longer match the current matrix structure.
  //   // Track the matrix pointer and dimensions/nnz so stale setup data is reset
  //   // before running SpMV with a different matrix.
  //   bool matrix_changed =
  //       (matrix_for_matvec_ != A) || (matvec_num_rows_ != A->getNumRows()) || (matvec_num_cols_ != A->getNumColumns()) || (matvec_nnz_ != A->getNnz());

  //   if (matrix_changed || values_changed_)
  //   {
  //     workspace_->resetMatvecSetup();
  //   }
  //   rocsparse_spmat_descr mat_A       = workspace_->getSpmvMatrixDescriptor();
  //   void* buffer_spmv = workspace_->getSpmvBuffer();
  //   if (!workspace_->matvecSetup())
  //   {
  //     // Setup, allocate, then compute.
  //     status = rocsparse_create_csr_descr(&mat_A,
  //                                         A->getNumRows(),
  //                                         A->getNumColumns(),
  //                                         A->getNnz(),
  //                                         A->getRowData(memory::DEVICE),
  //                                         A->getColData(memory::DEVICE),
  //                                         A->getValues(memory::DEVICE),
  //                                         rocsparse_indextype_i32,
  //                                         rocsparse_indextype_i32,
  //                                         rocsparse_index_base_zero,
  //                                         rocsparse_datatype_f64_r);
  //     error_sum += status;
  //     index_type bufferSize = 0;

  //     status = rocsparse_spmm(handle_cusparse,
  //                             rocsparse_operation_none,
  //                             rocsparse_operation_none,
  //                             &MINUS_ONE,
  //                             mat_A,
  //                             mat_X,
  //                             &ONE,
  //                             mat_AX,
  //                             rocsparse_datatype_f64_r,
  //                             rocsparse_spmm_alg_default,
  //                             rocsparse_spmm_stage_buffer_size,
  //                             &bufferSize,
  //                             nullptr);
  //     error_sum += status;
  //     mem_.allocateBufferOnDevice(&buffer_spmv, bufferSize);
  //     workspace_->setSpmvMatrixDescriptor(mat_A);
  //     workspace_->setSpmvBuffer(buffer_spmv, bufferSize);

  //     status = rocsparse_spmm(handle_cusparse,
  //                             rocsparse_operation_none,
  //                             rocsparse_operation_none,
  //                             &MINUS_ONE,
  //                             mat_A,
  //                             mat_X,
  //                             &ONE,
  //                             mat_AX,
  //                             rocsparse_datatype_f64_r,
  //                             rocsparse_spmm_alg_default,
  //                             rocsparse_spmm_stage_preprocess,
  //                             &bufferSize,
  //                             buffer_spmv);
  //     error_sum += status;

  //     workspace_->matvecSetupDone();

  //     matrix_for_matvec_ = A;
  //     matvec_num_rows_   = A->getNumRows();
  //     matvec_num_cols_   = A->getNumColumns();
  //     matvec_nnz_        = A->getNnz();

  //     values_changed_ = false;
  //   }

  //   index_type bufferSize = getSpmvBufferSize();

  //   status = rocsparse_spmm(handle_cusparse,
  //                           rocsparse_operation_none,
  //                           rocsparse_operation_none,
  //                           alpha,
  //                           mat_A,
  //                           mat_X,
  //                           beta,
  //                           mat_AX,
  //                           rocsparse_datatype_f64_r,
  //                           rocsparse_spmm_alg_default,
  //                           rocsparse_spmm_stage_compute,
  //                           &bufferSize,
  //                           buffer_spmv);
  //   error_sum += status;
  //   if (status)
  //     out::error() << "MatMultivec status: " << status << ". "
  //                  << "Last error code: " << mem_.getLastDeviceError() << ".\n";
  //   vec_result->setDataUpdated(memory::DEVICE);

  //   rocsparse_destroy_dnmat_descr(mat_X);
  //   rocsparse_destroy_dnmat_descr(mat_AX);
  //   return error_sum;
    return 1;
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
    hip::matrixRowSums(A->getNumRows(),
                       A->getNnz(),
                       A->getRowData(memory::DEVICE),
                       A->getValues(memory::DEVICE),
                       d_r);
    mem_.deviceSynchronize();

    hip::vectorInfNorm(A->getNumRows(),
                       d_r,
                       workspace_->getNormBuffer(),
                       norm);
    return 0;
  }

  real_type MatrixHandlerHip::norm(matrix::Sparse* A)
  {
    
    rocblas_handle handle_rocblas = workspace_->getRocblasHandle();

    double         nrm{0.0};
    rocblas_status st = rocblas_ddot(handle_rocblas,
                                   A->getNnz(),
                                   A->getValues(memory::DEVICE),
                                   1,
                                   A->getValues(memory::DEVICE),
                                   1,
                                   &nrm);
    if (st != 0)
    {
      out::error() << "matrix norm returned error code " << st << "\n";
    }
    return sqrt(nrm);
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
    // Ensure the shared transpose workspace is large enough for this matrix.
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
    error_sum += workspace_->setTransposeBufferWorkspace(bufferSize);
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

  // ...
  int MatrixHandlerHip::extractRootDiagonal(matrix::Csr* A, vector_type* diag)
  {
    real_type*  diag_data = diag->getData(memory::DEVICE);
    const index_type* a_row_ptr = A->getRowData(memory::DEVICE);
    const index_type* a_col_idx = A->getColData(memory::DEVICE);
    const real_type*  a_vals    = A->getValues(memory::DEVICE);
    index_type  n         = A->getNumRows();
    hip::extractRootDiagonal(n, a_row_ptr, a_col_idx, a_vals, diag_data);
    A->setUpdated(memory::DEVICE);
    return 0;
  }

  // ...
  int MatrixHandlerHip::extractInverseRootDiagonal(matrix::Csr* A, vector_type* diag)
  {
    real_type*  diag_data = diag->getData(memory::DEVICE);
    const index_type* a_row_ptr = A->getRowData(memory::DEVICE);
    const index_type* a_col_idx = A->getColData(memory::DEVICE);
    const real_type*  a_vals    = A->getValues(memory::DEVICE);
    index_type  n         = A->getNumRows();
    hip::extractInverseRootDiagonal(n, a_row_ptr, a_col_idx, a_vals, diag_data);
    A->setUpdated(memory::DEVICE);
    return 0;
  }

} // namespace ReSolve
