#include "LinSolverDirectRocSparseILU0.hpp"

#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using out = io::Logger;

  LinSolverDirectRocSparseILU0::LinSolverDirectRocSparseILU0(LinAlgWorkspaceHIP* workspace)
  {
    workspace_ = workspace;
    initParamList();
  }

  LinSolverDirectRocSparseILU0::~LinSolverDirectRocSparseILU0()
  {
    mem_.deleteOnDevice(d_aux1_);
    mem_.deleteOnDevice(d_ILU_vals_);
  }

  int LinSolverDirectRocSparseILU0::setup(matrix::Sparse* A,
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

    // set up descriptors

    // Create matrix descriptor for A
    rocsparse_create_mat_descr(&descr_A_);

    // Create matrix descriptor for L
    rocsparse_create_mat_descr(&descr_L_);
    rocsparse_set_mat_fill_mode(descr_L_, rocsparse_fill_mode_lower);
    rocsparse_set_mat_diag_type(descr_L_, rocsparse_diag_type_unit);

    // Create matrix descriptor for U
    rocsparse_create_mat_descr(&descr_U_);
    rocsparse_set_mat_fill_mode(descr_U_, rocsparse_fill_mode_upper);
    rocsparse_set_mat_diag_type(descr_U_, rocsparse_diag_type_non_unit);

    // Create matrix info structure
    rocsparse_create_mat_info(&info_A_);

    size_t buffer_size_A;
    size_t buffer_size_L;
    size_t buffer_size_U;

    status_rocsparse_ = rocsparse_dcsrilu0_buffer_size(workspace_->getRocsparseHandle(),
                                                       n,
                                                       nnz,
                                                       descr_A_,
                                                       d_ILU_vals_, // vals_,
                                                       A_->getRowData(ReSolve::memory::DEVICE),
                                                       A_->getColData(ReSolve::memory::DEVICE),
                                                       info_A_,
                                                       &buffer_size_A);
    if (status_rocsparse_ != 0)
    {
      io::Logger::warning() << "Buffer size estimate for ILU0 failed with code: " << status_rocsparse_ << " \n";
    }

    error_sum += status_rocsparse_;
    status_rocsparse_ = rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(),
                                                     rocsparse_operation_none,
                                                     n,
                                                     nnz,
                                                     descr_L_,
                                                     d_ILU_vals_, // vals_,
                                                     A_->getRowData(ReSolve::memory::DEVICE),
                                                     A_->getColData(ReSolve::memory::DEVICE),
                                                     info_A_,
                                                     &buffer_size_L);
    if (status_rocsparse_ != 0)
    {
      io::Logger::warning() << "Buffer size estimate for L solve failed with code: " << status_rocsparse_ << " \n";
    }

    error_sum += status_rocsparse_;

    status_rocsparse_ = rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(),
                                                     rocsparse_operation_none,
                                                     n,
                                                     nnz,
                                                     descr_U_,
                                                     d_ILU_vals_, // vals_,
                                                     A_->getRowData(ReSolve::memory::DEVICE),
                                                     A_->getColData(ReSolve::memory::DEVICE),
                                                     info_A_,
                                                     &buffer_size_U);
    if (status_rocsparse_ != 0)
    {
      io::Logger::warning() << "Buffer size estimate for U solve failed with code: " << status_rocsparse_ << " \n";
    }
    error_sum += status_rocsparse_;

    size_t buffer_size = std::max(buffer_size_A, std::max(buffer_size_L, buffer_size_U));

    mem_.allocateBufferOnDevice(&buffer_, buffer_size);

    // Now analysis
    status_rocsparse_ = rocsparse_dcsrilu0_analysis(workspace_->getRocsparseHandle(),
                                                    n,
                                                    nnz,
                                                    descr_A_,
                                                    d_ILU_vals_, // vals_,
                                                    A_->getRowData(ReSolve::memory::DEVICE),
                                                    A_->getColData(ReSolve::memory::DEVICE),
                                                    info_A_,
                                                    rocsparse_analysis_policy_reuse,
                                                    rocsparse_solve_policy_auto,
                                                    buffer_);

    if (status_rocsparse_ != 0)
    {
      io::Logger::warning() << "ILU0 decomposition analysis failed with code: " << status_rocsparse_ << " \n";
    }
    error_sum += status_rocsparse_;

    status_rocsparse_ = rocsparse_dcsrilu0_numeric_boost(workspace_->getRocsparseHandle(),
                                                         info_A_,
                                                         static_cast<int>(numeric_boost_),
                                                         &boost_tolerance_,
                                                         &boost_value_);

    if (status_rocsparse_ != rocsparse_status_success)
    {
      out::error() << "HIP ILU0 numerical boost setup failed with status " << status_rocsparse_ << "\n";
      return 1;
    }

    status_rocsparse_ = rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(),
                                                  rocsparse_operation_none,
                                                  n,
                                                  nnz,
                                                  descr_L_,
                                                  d_ILU_vals_, // vals_,
                                                  A_->getRowData(ReSolve::memory::DEVICE),
                                                  A_->getColData(ReSolve::memory::DEVICE),
                                                  info_A_,
                                                  rocsparse_analysis_policy_reuse,
                                                  rocsparse_solve_policy_auto,
                                                  buffer_);
    if (status_rocsparse_ != 0)
    {
      io::Logger::warning() << "Solve analysis for L solve failed with code: " << status_rocsparse_ << " \n";
    }
    error_sum += status_rocsparse_;

    status_rocsparse_ = rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(),
                                                  rocsparse_operation_none,
                                                  n,
                                                  nnz,
                                                  descr_U_,
                                                  d_ILU_vals_, // vals_,
                                                  A_->getRowData(ReSolve::memory::DEVICE),
                                                  A_->getColData(ReSolve::memory::DEVICE),
                                                  info_A_,
                                                  rocsparse_analysis_policy_reuse,
                                                  rocsparse_solve_policy_auto,
                                                  buffer_);
    if (status_rocsparse_ != 0)
    {
      io::Logger::warning() << "Solve analysis for U solve failed with code: " << status_rocsparse_ << " \n";
    }

    error_sum += status_rocsparse_;
    // allocate aux data

    // and now the actual decomposition

    // Compute incomplete LU factorization

    status_rocsparse_ = rocsparse_dcsrilu0(workspace_->getRocsparseHandle(),
                                           n,
                                           nnz,
                                           descr_A_,
                                           d_ILU_vals_, // vals_
                                           A_->getRowData(ReSolve::memory::DEVICE),
                                           A_->getColData(ReSolve::memory::DEVICE),
                                           info_A_,
                                           rocsparse_solve_policy_auto,
                                           buffer_);
    if (status_rocsparse_ != 0)
    {
      io::Logger::warning() << "ILU0 decomposition failed with code: " << status_rocsparse_ << " \n";
    }

    error_sum += status_rocsparse_;

    mem_.allocateArrayOnDevice(&d_aux1_, n);
    return error_sum;
  }

  int LinSolverDirectRocSparseILU0::reset(matrix::Sparse* A)
  {
    int error_sum  = 0;
    this->A_       = A;
    index_type n   = A_->getNumRows();
    index_type nnz = A_->getNnz();
    mem_.copyArrayDeviceToDevice(d_ILU_vals_, A_->getValues(ReSolve::memory::DEVICE), nnz);

    status_rocsparse_ = rocsparse_dcsrilu0(workspace_->getRocsparseHandle(),
                                           n,
                                           nnz,
                                           descr_A_,
                                           d_ILU_vals_, // vals_,
                                           A_->getRowData(ReSolve::memory::DEVICE),
                                           A_->getColData(ReSolve::memory::DEVICE),
                                           info_A_,
                                           rocsparse_solve_policy_auto,
                                           buffer_);

    error_sum += status_rocsparse_;

    return error_sum;
  }

  // solution is returned in RHS
  int LinSolverDirectRocSparseILU0::solve(vector_type* rhs)
  {
    int error_sum     = 0;
    status_rocsparse_ = rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                               rocsparse_operation_none,
                                               A_->getNumRows(),
                                               A_->getNnz(),
                                               &(constants::ONE),
                                               descr_L_,
                                               d_ILU_vals_, // vals_,
                                               A_->getRowData(ReSolve::memory::DEVICE),
                                               A_->getColData(ReSolve::memory::DEVICE),
                                               info_A_,
                                               rhs->getData(ReSolve::memory::DEVICE),
                                               d_aux1_, // result
                                               rocsparse_solve_policy_auto,
                                               buffer_);
    error_sum += status_rocsparse_;

    status_rocsparse_ = rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                               rocsparse_operation_none,
                                               A_->getNumRows(),
                                               A_->getNnz(),
                                               &(constants::ONE),
                                               descr_U_,
                                               d_ILU_vals_, // vals_,
                                               A_->getRowData(ReSolve::memory::DEVICE),
                                               A_->getColData(ReSolve::memory::DEVICE),
                                               info_A_,
                                               d_aux1_,                               // input
                                               rhs->getData(ReSolve::memory::DEVICE), // result
                                               rocsparse_solve_policy_auto,
                                               buffer_);
    error_sum += status_rocsparse_;
    rhs->setDataUpdated(ReSolve::memory::DEVICE);

    return error_sum;
  }

  int LinSolverDirectRocSparseILU0::solve(vector_type* rhs, vector_type* x)
  {
    int error_sum = 0;

    status_rocsparse_ = rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                               rocsparse_operation_none,
                                               A_->getNumRows(),
                                               A_->getNnz(),
                                               &(constants::ONE),
                                               descr_L_,
                                               d_ILU_vals_, // vals_,
                                               A_->getRowData(ReSolve::memory::DEVICE),
                                               A_->getColData(ReSolve::memory::DEVICE),
                                               info_A_,
                                               rhs->getData(ReSolve::memory::DEVICE),
                                               d_aux1_, // result
                                               rocsparse_solve_policy_auto,
                                               buffer_);
    error_sum += status_rocsparse_;

    status_rocsparse_ = rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                               rocsparse_operation_none,
                                               A_->getNumRows(),
                                               A_->getNnz(),
                                               &(constants::ONE),
                                               descr_U_,
                                               d_ILU_vals_, // vals_,
                                               A_->getRowData(ReSolve::memory::DEVICE),
                                               A_->getColData(ReSolve::memory::DEVICE),
                                               info_A_,
                                               d_aux1_,                             // input
                                               x->getData(ReSolve::memory::DEVICE), // result
                                               rocsparse_solve_policy_auto,
                                               buffer_);
    error_sum += status_rocsparse_;

    x->setDataUpdated(ReSolve::memory::DEVICE);

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
  int LinSolverDirectRocSparseILU0::setCliParam(const std::string id, const std::string value)
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
  std::string LinSolverDirectRocSparseILU0::getCliParamString(const std::string id) const
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
  index_type LinSolverDirectRocSparseILU0::getCliParamInt(const std::string id) const
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
  real_type LinSolverDirectRocSparseILU0::getCliParamReal(const std::string id) const
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
  bool LinSolverDirectRocSparseILU0::getCliParamBool(const std::string id) const
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
  int LinSolverDirectRocSparseILU0::printCliParam(const std::string id) const
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
  void LinSolverDirectRocSparseILU0::initParamList()
  {
    params_list_["numeric_boost"]   = NUMERIC_BOOST;
    params_list_["boost_tolerance"] = BOOST_TOLERANCE;
    params_list_["boost_value"]     = BOOST_VALUE;
  }

} // namespace ReSolve
