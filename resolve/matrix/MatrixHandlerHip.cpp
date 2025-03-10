#include <algorithm>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#include <resolve/hip/hipKernels.h>
#include "MatrixHandlerHip.hpp"

namespace ReSolve {
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
  int MatrixHandlerHip::matvec(matrix::Sparse* A, 
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
    rocsparse_status status;

    rocsparse_handle handle_rocsparse = workspace_->getRocsparseHandle();
    
    rocsparse_mat_info infoA = workspace_->getSpmvMatrixInfo();
    rocsparse_mat_descr descrA =  workspace_->getSpmvMatrixDescriptor();
    
    if (!workspace_->matvecSetup()) {
      //setup first, allocate, etc.
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
                                          A->getValues( memory::DEVICE), 
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
                              A->getValues( memory::DEVICE), 
                              A->getRowData(memory::DEVICE),
                              A->getColData(memory::DEVICE),
                              infoA,
                              vec_x->getData(memory::DEVICE),
                              beta,
                              vec_result->getData(memory::DEVICE));

    error_sum += status;
    mem_.deviceSynchronize();
    if (status) {
      out::error() << "Matvec status: "   << status                    << ". "
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
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
           "Matrix has to be in CSR format for matrix-vector product.\n");

    real_type* d_r = workspace_->getDr();
    index_type d_r_size = workspace_->getDrSize();
    
    if (d_r_size != A->getNumRows()) {
      if (d_r_size != 0) {
        mem_.deleteOnDevice(d_r);
      }
      mem_.allocateArrayOnDevice(&d_r, A->getNumRows());
      workspace_->setDrSize(A->getNumRows());
      workspace_->setDr(d_r);
    }
    
    if (workspace_->getNormBufferState() == false) { // not allocated  
      real_type* buffer;
      mem_.allocateArrayOnDevice(&buffer, 1024);
      workspace_->setNormBuffer(buffer);
      workspace_->setNormBufferState(true);
    }

    mem_.deviceSynchronize();
    matrix_row_sums(A->getNumRows(),
                    A->getNnz(),
                    A->getRowData(memory::DEVICE),
                    A->getValues(memory::DEVICE),
                    d_r);
    mem_.deviceSynchronize();

    vector_inf_norm(A->getNumRows(),  
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
    index_type m = A_csc->getNumColumns();
    index_type n = A_csc->getNumRows();
    index_type nnz = A_csc->getNnz();
    size_t bufferSize;
    void* d_work;

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
                                A_csc->getValues( memory::DEVICE), 
                                A_csc->getColData(memory::DEVICE), 
                                A_csc->getRowData(memory::DEVICE), 
                                A_csr->getValues( memory::DEVICE), 
                                A_csr->getColData(memory::DEVICE),
                                A_csr->getRowData(memory::DEVICE), 
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                d_work);
    error_sum += status;
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
   * @param[out]      At - Transposed matrix
   * @param[in]       allocated - flag indicating if At is already allocated
   * 
   * @return int error_sum, 0 if successful
   *
   * @warning This method works only for `real_type == double`.
   */
  int MatrixHandlerHip::transpose(matrix::Csr* A, matrix::Csr* At, bool allocated)
  {
    index_type error_sum = 0;
    index_type m = A->getNumRows();
    index_type n = A->getNumColumns();
    index_type nnz = A->getNnz();
    rocsparse_status status;
    if (!allocated) {
      // check dimensions of A and At
      assert(A->getNumRows() == At->getNumColumns() && "Number of rows in A must be equal to number of columns in At");
      assert(A->getNumColumns() == At->getNumRows() && "Number of columns in A must be equal to number of rows in At");

      At->allocateMatrixData(memory::DEVICE);

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
      mem_.allocateBufferOnDevice(&transpose_workspace_, bufferSize);
    }
    status = rocsparse_dcsr2csc(workspace_->getRocsparseHandle(),
                                m,
                                n,
                                nnz,
                                A->getValues( memory::DEVICE), 
                                A->getRowData(memory::DEVICE), 
                                A->getColData(memory::DEVICE), 
                                At->getValues( memory::DEVICE), 
                                At->getColData(memory::DEVICE),
                                At->getRowData(memory::DEVICE), 
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                transpose_workspace_);
    error_sum += status;
    // Values on the device are updated now -- mark them as such!
    At->setUpdated(memory::DEVICE);

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
  int MatrixHandlerHip::addConstantToNonzeroValues(matrix::Sparse* A, real_type alpha)
  {
    real_type* values = A->getValues(memory::DEVICE);
    index_type nnz = A->getNnz();
    hipAddConst(values, alpha, nnz);
    return 0;
  }

} // namespace ReSolve
