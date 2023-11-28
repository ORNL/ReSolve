#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include "LinSolverDirectCuSparseILU0.hpp"
#include <algorithm>
namespace ReSolve 
{
  LinSolverDirectCuSparseILU0::LinSolverDirectCuSparseILU0(LinAlgWorkspaceCUDA* workspace)
  {
    workspace_ = workspace;
  }

  LinSolverDirectCuSparseILU0::~LinSolverDirectCuSparseILU0()
  {
    mem_.deleteOnDevice(d_aux1_);
    mem_.deleteOnDevice(d_ILU_vals_);
    mem_.deleteOnDevice(buffer_);
    mem_.deleteOnDevice(buffer_LU_);
    cusparseDestroyMatDescr(descr_A_);
    cusparseDestroyMatDescr(descr_L_);
    cusparseDestroyMatDescr(descr_U_);
    cusparseDestroyCsrilu02Info(info_A_);
    cusparseDestroyCsrsv2Info(info_L_);
    cusparseDestroyCsrsv2Info(info_U_);
  }

  int LinSolverDirectCuSparseILU0::setup(matrix::Sparse* A,
                                         matrix::Sparse*,
                                         matrix::Sparse*,
                                         index_type*,
                                         index_type*,
                                         vector_type* rhs)
  {
    //remember - P and Q are generally CPU variables
    int error_sum = 0;
    this->A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();

    index_type nnz = A_->getNnzExpanded();
    printf("nnz = %d \n", nnz);
    mem_.allocateArrayOnDevice(&d_ILU_vals_,nnz);

    //copy A values to a buffer first
    mem_.copyArrayDeviceToDevice(d_ILU_vals_, A_->getValues(ReSolve::memory::DEVICE), nnz);

    mem_.allocateArrayOnDevice(&d_aux1_,n); 

    //set up descriptors

    // Create matrix descriptor for A
    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);

    // Create matrix descriptor for L and U
    cusparseCreateMatDescr(&descr_L_);
    cusparseSetMatIndexBase(descr_L_, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L_, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L_, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U_);
    cusparseSetMatIndexBase(descr_U_, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U_, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U_, CUSPARSE_DIAG_TYPE_NON_UNIT);


    // Create matrix info structure
    cusparseCreateCsrilu02Info(&info_A_);
    cusparseCreateCsrsv2Info(&info_L_);
    cusparseCreateCsrsv2Info(&info_U_);
    int buffer_size_A;
    int buffer_size_L;
    int buffer_size_U;

    status_cusparse_ = cusparseDcsrilu02_bufferSize(workspace_->getCusparseHandle(), 
                                                    n, 
                                                    nnz, 
                                                    descr_A_,
                                                    d_ILU_vals_, //vals_, 
                                                    A_->getRowData(ReSolve::memory::DEVICE), 
                                                    A_->getColData(ReSolve::memory::DEVICE), 
                                                    info_A_, 
                                                    &buffer_size_A);

    mem_.allocateBufferOnDevice(&buffer_, (size_t) buffer_size_A);
    error_sum += status_cusparse_;
    // Now analysis
    status_cusparse_ = cusparseDcsrilu02_analysis(workspace_->getCusparseHandle(), 
                                                  n, 
                                                  nnz, 
                                                  descr_A_,
                                                  d_ILU_vals_, //vals_, 
                                                  A_->getRowData(ReSolve::memory::DEVICE), 
                                                  A_->getColData(ReSolve::memory::DEVICE), 
                                                  info_A_,
                                                  CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                  buffer_);

    error_sum += status_cusparse_;

    // Compute incomplete LU factorization
    status_cusparse_ = cusparseDcsrilu02(workspace_->getCusparseHandle(), 
                                         n, 
                                         nnz, 
                                         descr_A_,
                                         d_ILU_vals_, //vals_ 
                                         A_->getRowData(ReSolve::memory::DEVICE), 
                                         A_->getColData(ReSolve::memory::DEVICE), 
                                         info_A_,
                                         CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                         buffer_);


    error_sum += status_cusparse_;

    // now take care of LU solve 

    status_cusparse_ = cusparseDcsrsv2_bufferSize(workspace_->getCusparseHandle(), 
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                  n,
                                                  nnz,
                                                  descr_L_, 
                                                  d_ILU_vals_, //vals_ 
                                                  A_->getRowData(ReSolve::memory::DEVICE), 
                                                  A_->getColData(ReSolve::memory::DEVICE), 
                                                  info_L_, 
                                                  &buffer_size_L);
    error_sum += status_cusparse_;

    status_cusparse_ = cusparseDcsrsv2_bufferSize(workspace_->getCusparseHandle(), 
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                  n,
                                                  nnz,
                                                  descr_U_, 
                                                  d_ILU_vals_, //vals_ 
                                                  A_->getRowData(ReSolve::memory::DEVICE), 
                                                  A_->getColData(ReSolve::memory::DEVICE), 
                                                  info_U_, 
                                                  &buffer_size_U);
    error_sum += status_cusparse_;
    size_t buffer_size_LU;
    if (buffer_size_L > buffer_size_U) {
      buffer_size_LU = buffer_size_L;
    } else {
      buffer_size_LU = buffer_size_U;
    }
    mem_.allocateBufferOnDevice(&buffer_LU_, buffer_size_LU);

    status_cusparse_ = cusparseDcsrsv2_analysis(workspace_->getCusparseHandle(), 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                n,
                                                nnz,
                                                descr_L_,
                                                d_ILU_vals_, //vals_ 
                                                A_->getRowData(ReSolve::memory::DEVICE), 
                                                A_->getColData(ReSolve::memory::DEVICE),
                                                info_L_,
                                                CUSPARSE_SOLVE_POLICY_USE_LEVEL, 
                                                buffer_LU_);
    error_sum += status_cusparse_;


    status_cusparse_ = cusparseDcsrsv2_analysis(workspace_->getCusparseHandle(), 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                n,
                                                nnz,
                                                descr_U_,
                                                d_ILU_vals_, //vals_ 
                                                A_->getRowData(ReSolve::memory::DEVICE), 
                                                A_->getColData(ReSolve::memory::DEVICE),
                                                info_U_,
                                                CUSPARSE_SOLVE_POLICY_USE_LEVEL, 
                                                buffer_LU_);

    error_sum += status_cusparse_;
    return error_sum;
  }

  int LinSolverDirectCuSparseILU0::reset(matrix::Sparse* A)
  {
    int error_sum = 0;
    this->A_ = A;
    index_type n = A_->getNumRows();
    index_type nnz = A_->getNnzExpanded();
    mem_.copyArrayDeviceToDevice(d_ILU_vals_, A_->getValues(ReSolve::memory::DEVICE), nnz);

    status_cusparse_ = cusparseDcsrilu02(workspace_->getCusparseHandle(), 
                                         n, 
                                         nnz, 
                                         descr_A_,
                                         d_ILU_vals_, //vals_ 
                                         A_->getRowData(ReSolve::memory::DEVICE), 
                                         A_->getColData(ReSolve::memory::DEVICE), 
                                         info_A_,
                                         CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                         buffer_);
    //rerun tri solve analysis - to be updated
    error_sum += status_cusparse_;

    return error_sum;
  }
  // solution is returned in RHS
  int LinSolverDirectCuSparseILU0::solve(vector_type* rhs)
  {

    int error_sum = 0;
    /*cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L, // replace with cusparseSpSV
      d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
      d_x, d_z, policy_L, pBuffer);
     * */
    status_cusparse_ = cusparseDcsrsv2_solve(workspace_->getCusparseHandle(), 
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             A_->getNumRows(),
                                             A_->getNnzExpanded(),
                                             &(constants::ONE), 
                                             descr_L_,
                                             d_ILU_vals_, //vals_ 
                                             A_->getRowData(ReSolve::memory::DEVICE), 
                                             A_->getColData(ReSolve::memory::DEVICE),
                                             info_L_,
                                             rhs->getData(ReSolve::memory::DEVICE),
                                             d_aux1_, 
                                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, 
                                             buffer_LU_);
    error_sum += status_cusparse_;

    status_cusparse_ = cusparseDcsrsv2_solve(workspace_->getCusparseHandle(), 
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             A_->getNumRows(),
                                             A_->getNnzExpanded(),
                                             &(constants::ONE), 
                                             descr_U_,
                                             d_ILU_vals_, //vals_ 
                                             A_->getRowData(ReSolve::memory::DEVICE), 
                                             A_->getColData(ReSolve::memory::DEVICE),
                                             info_U_,
                                             d_aux1_,
                                             rhs->getData(ReSolve::memory::DEVICE),
                                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, 
                                             buffer_LU_);
    error_sum += status_cusparse_;
    rhs->setDataUpdated(ReSolve::memory::DEVICE);

    return error_sum;
  }

  int LinSolverDirectCuSparseILU0::solve(vector_type* rhs, vector_type* x)
  {
    int error_sum = 0;
    status_cusparse_ = cusparseDcsrsv2_solve(workspace_->getCusparseHandle(), 
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             A_->getNumRows(),
                                             A_->getNnzExpanded(),
                                             &(constants::ONE), 
                                             descr_L_,
                                             d_ILU_vals_, //vals_ 
                                             A_->getRowData(ReSolve::memory::DEVICE), 
                                             A_->getColData(ReSolve::memory::DEVICE),
                                             info_L_,
                                             rhs->getData(ReSolve::memory::DEVICE),
                                             d_aux1_, 
                                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, 
                                             buffer_LU_);
    error_sum += status_cusparse_;

    status_cusparse_ = cusparseDcsrsv2_solve(workspace_->getCusparseHandle(), 
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             A_->getNumRows(),
                                             A_->getNnzExpanded(),
                                             &(constants::ONE), 
                                             descr_U_,
                                             d_ILU_vals_, //vals_ 
                                             A_->getRowData(ReSolve::memory::DEVICE), 
                                             A_->getColData(ReSolve::memory::DEVICE),
                                             info_U_,
                                             d_aux1_,
                                             x->getData(ReSolve::memory::DEVICE),
                                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, 
                                             buffer_LU_);
    error_sum += status_cusparse_;
    x->setDataUpdated(ReSolve::memory::DEVICE);

    return error_sum;
  }
}// namespace resolve
