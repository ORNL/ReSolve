#include "LinSolverDirectCuSparseILU0.hpp"

#include <algorithm>

#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using out = io::Logger;

  LinSolverDirectCuSparseILU0::LinSolverDirectCuSparseILU0(LinAlgWorkspaceCUDA* workspace)
  {
    workspace_ = workspace;
    initParamList();
  }

  LinSolverDirectCuSparseILU0::~LinSolverDirectCuSparseILU0()
  {
    mem_.deleteOnDevice(d_aux1_);
    mem_.deleteOnDevice(d_aux2_);
    mem_.deleteOnDevice(d_ILU_vals_);
  }

  int LinSolverDirectCuSparseILU0::setup(matrix::Sparse* A,
                                         matrix::Sparse*,
                                         matrix::Sparse*,
                                         index_type*,
                                         index_type*,
                                         vector_type*)
  {
    // remember - P and Q are generally CPU variables
    int error_sum = 0;
    this->A_      = (matrix::Csr*) A;
    index_type n  = A_->getNumRows();

    index_type nnz = A_->getNnz();
    mem_.allocateArrayOnDevice(&d_ILU_vals_, nnz);
    // copy A values to a buffer first
    mem_.copyArrayDeviceToDevice(d_ILU_vals_, A_->getValues(ReSolve::memory::DEVICE), nnz);

    mem_.allocateArrayOnDevice(&d_aux1_, n);
    mem_.allocateArrayOnDevice(&d_aux2_, n);
    cudaMemset(d_aux1_, 1, n * sizeof(double));
    cusparseCreateDnVec(&vec_X_, n, d_aux1_, CUDA_R_64F);
    cusparseCreateDnVec(&vec_Y_, n, d_aux2_, CUDA_R_64F);

    // set up descriptors

    // Create matrix descriptor for A
    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);

    // Create matrix descriptor for L and U
    cusparseSpSV_createDescr(&descr_spsv_L_);
    cusparseSpSV_createDescr(&descr_spsv_U_);

    cusparseCreateCsr(&mat_L_,
                      n,
                      n,
                      nnz,
                      A_->getRowData(ReSolve::memory::DEVICE),
                      A_->getColData(ReSolve::memory::DEVICE),
                      d_ILU_vals_, // vals_,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_64F);

    cusparseCreateCsr(&mat_U_,
                      n,
                      n,
                      nnz,
                      A_->getRowData(ReSolve::memory::DEVICE),
                      A_->getColData(ReSolve::memory::DEVICE),
                      d_ILU_vals_, // vals_,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_64F);

    // Create matrix info structure
    cusparseCreateCsrilu02Info(&info_A_);

    int    buffer_size_A;
    size_t buffer_size_L;
    size_t buffer_size_U;

    status_cusparse_ = cusparseDcsrilu02_bufferSize(workspace_->getCusparseHandle(),
                                                    n,
                                                    nnz,
                                                    descr_A_,
                                                    d_ILU_vals_, // vals_,
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
                                                  d_ILU_vals_, // vals_,
                                                  A_->getRowData(ReSolve::memory::DEVICE),
                                                  A_->getColData(ReSolve::memory::DEVICE),
                                                  info_A_,
                                                  CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                  buffer_);

    error_sum += status_cusparse_;

    status_cusparse_ = cusparseDcsrilu02_numericBoost(workspace_->getCusparseHandle(),
                                                      info_A_,
                                                      static_cast<int>(numeric_boost_),
                                                      &boost_tolerance_,
                                                      &boost_value_);

    if (status_cusparse_ != CUSPARSE_STATUS_SUCCESS)
    {
      out::error() << "CUDA ILU0 numerical boost setup failed with status "
                   << static_cast<int>(status_cusparse_) << "\n";
      return 1;
    }

    // and now the actual decomposition

    // Compute incomplete LU factorization
    status_cusparse_ = cusparseDcsrilu02(workspace_->getCusparseHandle(),
                                         n,
                                         nnz,
                                         descr_A_,
                                         d_ILU_vals_, // vals_
                                         A_->getRowData(ReSolve::memory::DEVICE),
                                         A_->getColData(ReSolve::memory::DEVICE),
                                         info_A_,
                                         CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                         buffer_);

    error_sum += status_cusparse_;

    // now take care of LU solve

    // now create actual Sparse matrix  OBJECTS for L and U

    cusparseFillMode_t fillmodeL = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t fillmodeU = CUSPARSE_FILL_MODE_UPPER;

    cusparseDiagType_t diagtypeL = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t diagtypeU = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseSpMatSetAttribute(mat_L_,
                              CUSPARSE_SPMAT_FILL_MODE,
                              &fillmodeL,
                              sizeof(fillmodeL));

    cusparseSpMatSetAttribute(mat_U_,
                              CUSPARSE_SPMAT_FILL_MODE,
                              &fillmodeU,
                              sizeof(fillmodeU));

    cusparseSpMatSetAttribute(mat_L_,
                              CUSPARSE_SPMAT_DIAG_TYPE,
                              &diagtypeL,
                              sizeof(diagtypeL));

    cusparseSpMatSetAttribute(mat_U_,
                              CUSPARSE_SPMAT_DIAG_TYPE,
                              &diagtypeU,
                              sizeof(diagtypeU));

    status_cusparse_ = cusparseSpSV_bufferSize(workspace_->getCusparseHandle(),
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &(constants::ONE),
                                               mat_L_,
                                               vec_X_,
                                               vec_Y_,
                                               CUDA_R_64F,
                                               CUSPARSE_SPSV_ALG_DEFAULT,
                                               descr_spsv_L_,
                                               &buffer_size_L);
    error_sum += status_cusparse_;

    mem_.allocateBufferOnDevice(&buffer_L_, buffer_size_L);
    status_cusparse_ = cusparseSpSV_bufferSize(workspace_->getCusparseHandle(),
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &(constants::ONE),
                                               mat_U_,
                                               vec_X_,
                                               vec_Y_,
                                               CUDA_R_64F,
                                               CUSPARSE_SPSV_ALG_DEFAULT,
                                               descr_spsv_U_,
                                               &buffer_size_U);
    error_sum += status_cusparse_;

    mem_.allocateBufferOnDevice(&buffer_U_, buffer_size_U);

    status_cusparse_ = cusparseSpSV_analysis(workspace_->getCusparseHandle(),
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &(constants::ONE),
                                             mat_L_,
                                             vec_X_,
                                             vec_Y_,
                                             CUDA_R_64F,
                                             CUSPARSE_SPSV_ALG_DEFAULT,
                                             descr_spsv_L_,
                                             buffer_L_);
    error_sum += status_cusparse_;

    status_cusparse_ = cusparseSpSV_analysis(workspace_->getCusparseHandle(),
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &(constants::ONE),
                                             mat_U_,
                                             vec_X_,
                                             vec_Y_,
                                             CUDA_R_64F,
                                             CUSPARSE_SPSV_ALG_DEFAULT,
                                             descr_spsv_U_,
                                             buffer_U_);

    error_sum += status_cusparse_;
    cusparseDestroyDnVec(vec_X_);
    cusparseDestroyDnVec(vec_Y_);
    return error_sum;
  }

  int LinSolverDirectCuSparseILU0::reset(matrix::Sparse* A)
  {
    int error_sum  = 0;
    this->A_       = A;
    index_type n   = A_->getNumRows();
    index_type nnz = A_->getNnz();
    mem_.copyArrayDeviceToDevice(d_ILU_vals_, A_->getValues(ReSolve::memory::DEVICE), nnz);

    status_cusparse_ = cusparseDcsrilu02(workspace_->getCusparseHandle(),
                                         n,
                                         nnz,
                                         descr_A_,
                                         d_ILU_vals_, // vals_
                                         A_->getRowData(ReSolve::memory::DEVICE),
                                         A_->getColData(ReSolve::memory::DEVICE),
                                         info_A_,
                                         CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                         buffer_);
    // rerun tri solve analysis - to be updated
    error_sum += status_cusparse_;

    return error_sum;
  }

  // solution is returned in RHS
  int LinSolverDirectCuSparseILU0::solve(vector_type* rhs)
  {
    int error_sum = 0;

    cusparseCreateDnVec(&vec_X_, A_->getNumRows(), rhs->getData(ReSolve::memory::DEVICE), CUDA_R_64F);
    cusparseCreateDnVec(&vec_Y_, A_->getNumRows(), d_aux1_, CUDA_R_64F);

    status_cusparse_ = cusparseSpSV_solve(workspace_->getCusparseHandle(),
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &(constants::ONE),
                                          mat_L_,
                                          vec_X_,
                                          vec_Y_,
                                          CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT,
                                          descr_spsv_L_);
    error_sum += status_cusparse_;

    status_cusparse_ = cusparseSpSV_solve(workspace_->getCusparseHandle(),
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &(constants::ONE),
                                          mat_U_,
                                          vec_Y_,
                                          vec_X_,
                                          CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT,
                                          descr_spsv_U_);
    error_sum += status_cusparse_;

    rhs->setDataUpdated(ReSolve::memory::DEVICE);

    cusparseDestroyDnVec(vec_X_);
    cusparseDestroyDnVec(vec_Y_);

    return error_sum;
  }

  int LinSolverDirectCuSparseILU0::solve(vector_type* rhs, vector_type* x)
  {
    int error_sum = 0;

    cusparseCreateDnVec(&vec_X_, A_->getNumRows(), rhs->getData(ReSolve::memory::DEVICE), CUDA_R_64F);
    cusparseCreateDnVec(&vec_Y_, A_->getNumRows(), d_aux1_, CUDA_R_64F);

    status_cusparse_ = cusparseSpSV_solve(workspace_->getCusparseHandle(),
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &(constants::ONE),
                                          mat_L_,
                                          vec_X_,
                                          vec_Y_,
                                          CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT,
                                          descr_spsv_L_);
    error_sum += status_cusparse_;

    cusparseCreateDnVec(&vec_X_, A_->getNumRows(), x->getData(ReSolve::memory::DEVICE), CUDA_R_64F);
    status_cusparse_ = cusparseSpSV_solve(workspace_->getCusparseHandle(),
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &(constants::ONE),
                                          mat_U_,
                                          vec_Y_,
                                          vec_X_,
                                          CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT,
                                          descr_spsv_U_);
    error_sum += status_cusparse_;

    x->setDataUpdated(ReSolve::memory::DEVICE);

    cusparseDestroyDnVec(vec_X_);
    cusparseDestroyDnVec(vec_Y_);

    return error_sum;
  }

  /**
   * @brief Set Cli parameters for ILU0 solver.
   *
   * @param[in] id    - string ID for parameter to set
   * @param[in] value - string value for parameter to set
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuSparseILU0::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
    case NUMERIC_BOOST:
      numeric_boost_ = (value == "yes");
      return 0;
    case BOOST_TOLERANCE:
      boost_tolerance_ = atof(value.c_str());
      return 0;
    case BOOST_VALUE:
      boost_value_ = atof(value.c_str());
      return 0;
    default:
      std::cout << "Setting parameter failed!\n";
      return 1;
    }
  }

  /**
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return std::string Value of the string parameter to return.
   */
  std::string LinSolverDirectCuSparseILU0::getCliParamString(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown string parameter " << id << "\n";
    }
    return "";
  }

  /**
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return int Value of the int parameter to return.
   */
  index_type LinSolverDirectCuSparseILU0::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown integer parameter " << id << "\n";
    }
    return -1;
  }

  /**
   * @brief Get the real parameter for the ILU0 solver.
   *
   * @param[in] id - string ID for parameter to get
   *
   * @return real_type - parameter value, or NaN if parameter is unknown
   */
  real_type LinSolverDirectCuSparseILU0::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
    case BOOST_TOLERANCE:
      return boost_tolerance_;
    case BOOST_VALUE:
      return boost_value_;
    default:
      out::error() << "Trying to get unknown real parameter " << id << "\n";
    }
    return std::numeric_limits<real_type>::quiet_NaN();
  }

  /**
   * @brief Get the boolean parameter for the ILU0 solver.
   *
   * @param[in] id - string ID for parameter to get
   *
   * @return bool - true if numeric boost is enabled, false otherwise
   */
  bool LinSolverDirectCuSparseILU0::getCliParamBool(const std::string id) const
  {
    switch (getParamId(id))
    {
    case NUMERIC_BOOST:
      return numeric_boost_;
    default:
      out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    }
    return false;
  }

  /**
   * @brief Print the ILU0 Cli parameters.
   *
   * @param[in] id - string ID for parameter to print
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuSparseILU0::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    case NUMERIC_BOOST:
      std::cout << numeric_boost_ << "\n";
      break;
    case BOOST_TOLERANCE:
      std::cout << boost_tolerance_ << "\n";
      break;
    case BOOST_VALUE:
      std::cout << boost_value_ << "\n";
      break;
    default:
      out::error() << "Trying to print unknown parameter " << id << "\n";
      return 1;
    }
    return 0;
  }

  /**
   * @brief Initialize the parameter list for ILU0 solver.
   *
   * @post params_list_ is populated with the ILU0 solver parameters:
   * - numeric_boost
   * - boost_tolerance
   * - boost_value
   */
  void LinSolverDirectCuSparseILU0::initParamList()
  {
    params_list_["numeric_boost"]   = NUMERIC_BOOST;
    params_list_["boost_tolerance"] = BOOST_TOLERANCE;
    params_list_["boost_value"]     = BOOST_VALUE;
  }

} // namespace ReSolve
