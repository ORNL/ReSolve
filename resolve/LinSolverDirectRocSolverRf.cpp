#include "LinSolverDirectRocSolverRf.hpp"

#include <resolve/hip/hipKernels.h>

#include <resolve/Profiling.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief Constructor for LinSolverDirectRocSolverRf
   *
   * @param[in] workspace - pointer to LinAlgWorkspaceHIP
   *
   * @post solve_mode_ is set to 1 or -1
   */
  LinSolverDirectRocSolverRf::LinSolverDirectRocSolverRf(LinAlgWorkspaceHIP* workspace)
  {
    workspace_  = workspace;
    infoM_      = nullptr;
    solve_mode_ = 1; // solve mode - 1: use rocsparse trisolve
    initParamList();
  }

  /**
   * @brief Destructor for LinSolverDirectRocSolverRf
   *
   * @pre d_P_, d_Q_, d_aux1_, d_aux2_, L_csr_, U_csr_, allocated on device
   *
   * @post All memory allocated on the device is freed
   */
  LinSolverDirectRocSolverRf::~LinSolverDirectRocSolverRf()
  {
    mem_.deleteOnDevice(d_P_);
    mem_.deleteOnDevice(d_Q_);

    mem_.deleteOnDevice(d_aux1_);
    mem_.deleteOnDevice(d_aux2_);

    delete L_csr_;
    delete U_csr_;
  }

  /**
   * @brief Setup function for LinSolverDirectRocSolverRf where factors are already in csr
   *
   * @param[in] A - matrix::Sparse* - matrix to solve
   * @param[in] L - matrix::Sparse* - lower triangular factor
   * @param[in] U - matrix::Sparse* - upper triangular factor
   * @param[in] P - index_type* - permutation vector P
   * @param[in] Q - index_type* - permutation vector Q
   * @param[in] rhs - vector_type* - right hand side
   *
   * @param[out] error_sum - int - sum of errors from setup
   */
  int LinSolverDirectRocSolverRf::setup(matrix::Sparse* A,
                                           matrix::Sparse* L,
                                           matrix::Sparse* U,
                                           index_type*     P,
                                           index_type*     Q,
                                           vector_type*    rhs)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);

    int error_sum = 0;

    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for rocsolverRf.\n");
    A_           = A;
    index_type n = A_->getNumRows();

    // set matrix info
    rocsolver_create_rfinfo(&infoM_, workspace_->getRocblasHandle());

    // Combine factors L and U into matrix M_
    combineFactors(L, U);

    M_->setUpdated(ReSolve::memory::HOST);
    M_->syncData(ReSolve::memory::DEVICE);

    // remember - P and Q are generally CPU variables
    if (d_P_ == nullptr)
    {
      mem_.allocateArrayOnDevice(&d_P_, n);
    }

    if (d_Q_ == nullptr)
    {
      mem_.allocateArrayOnDevice(&d_Q_, n);
    }
    mem_.copyArrayHostToDevice(d_P_, P, n);
    mem_.copyArrayHostToDevice(d_Q_, Q, n);

    mem_.deviceSynchronize();
    status_rocblas_ = rocsolver_dcsrrf_analysis(workspace_->getRocblasHandle(),
                                                n,
                                                1,
                                                A_->getNnz(),
                                                A_->getRowData(ReSolve::memory::DEVICE), // kRowPtr_,
                                                A_->getColData(ReSolve::memory::DEVICE), // jCol_,
                                                A_->getValues(ReSolve::memory::DEVICE),  // vals_,
                                                M_->getNnz(),
                                                M_->getRowData(ReSolve::memory::DEVICE),
                                                M_->getColData(ReSolve::memory::DEVICE),
                                                M_->getValues(ReSolve::memory::DEVICE), // vals_,
                                                d_P_,
                                                d_Q_,
                                                rhs->getData(ReSolve::memory::DEVICE),
                                                n,
                                                infoM_);

    mem_.deviceSynchronize();
    error_sum += status_rocblas_;
    // tri solve setup
    if (solve_mode_ == 1)
    { // OBSOLETE -- to be removed. Formerly known as "fast mode" TODO

      if (L_csr_ != nullptr)
      {
        delete L_csr_;
      }

      L_csr_ = new ReSolve::matrix::Csr(L->getNumRows(), L->getNumColumns(), L->getNnz());
      L_csr_->allocateMatrixData(ReSolve::memory::DEVICE);

      if (U_csr_ != nullptr)
      {
        delete U_csr_;
      }

      U_csr_ = new ReSolve::matrix::Csr(U->getNumRows(), U->getNumColumns(), U->getNnz());
      U_csr_->allocateMatrixData(ReSolve::memory::DEVICE);

      rocsparse_create_mat_descr(&(descr_L_));
      rocsparse_set_mat_fill_mode(descr_L_, rocsparse_fill_mode_lower);
      rocsparse_set_mat_index_base(descr_L_, rocsparse_index_base_zero);

      rocsparse_create_mat_descr(&(descr_U_));
      rocsparse_set_mat_index_base(descr_U_, rocsparse_index_base_zero);
      rocsparse_set_mat_fill_mode(descr_U_, rocsparse_fill_mode_upper);

      rocsparse_create_mat_info(&info_L_);
      rocsparse_create_mat_info(&info_U_);

      // local variables
      size_t L_buffer_size;
      size_t U_buffer_size;

      status_rocblas_ = rocsolver_dcsrrf_splitlu(workspace_->getRocblasHandle(),
                                                 n,
                                                 M_->getNnz(),
                                                 M_->getRowData(ReSolve::memory::DEVICE),
                                                 M_->getColData(ReSolve::memory::DEVICE),
                                                 M_->getValues(ReSolve::memory::DEVICE), // vals_,
                                                 L_csr_->getRowData(ReSolve::memory::DEVICE),
                                                 L_csr_->getColData(ReSolve::memory::DEVICE),
                                                 L_csr_->getValues(ReSolve::memory::DEVICE), // vals_,
                                                 U_csr_->getRowData(ReSolve::memory::DEVICE),
                                                 U_csr_->getColData(ReSolve::memory::DEVICE),
                                                 U_csr_->getValues(ReSolve::memory::DEVICE));

      error_sum += status_rocblas_;

      status_rocsparse_ = rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(),
                                                       rocsparse_operation_none,
                                                       n,
                                                       L_csr_->getNnz(),
                                                       descr_L_,
                                                       L_csr_->getValues(ReSolve::memory::DEVICE),
                                                       L_csr_->getRowData(ReSolve::memory::DEVICE),
                                                       L_csr_->getColData(ReSolve::memory::DEVICE),
                                                       info_L_,
                                                       &L_buffer_size);
      error_sum += status_rocsparse_;
      mem_.allocateBufferOnDevice(&L_buffer_, L_buffer_size);

      status_rocsparse_ = rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(),
                                                       rocsparse_operation_none,
                                                       n,
                                                       U_csr_->getNnz(),
                                                       descr_U_,
                                                       U_csr_->getValues(ReSolve::memory::DEVICE),
                                                       U_csr_->getRowData(ReSolve::memory::DEVICE),
                                                       U_csr_->getColData(ReSolve::memory::DEVICE),
                                                       info_U_,
                                                       &U_buffer_size);
      error_sum += status_rocsparse_;
      mem_.allocateBufferOnDevice(&U_buffer_, U_buffer_size);

      status_rocsparse_ = rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(),
                                                    rocsparse_operation_none,
                                                    n,
                                                    L_csr_->getNnz(),
                                                    descr_L_,
                                                    L_csr_->getValues(ReSolve::memory::DEVICE),
                                                    L_csr_->getRowData(ReSolve::memory::DEVICE),
                                                    L_csr_->getColData(ReSolve::memory::DEVICE),
                                                    info_L_,
                                                    rocsparse_analysis_policy_force,
                                                    rocsparse_solve_policy_auto,
                                                    L_buffer_);

      error_sum += status_rocsparse_;
      if (status_rocsparse_ != 0)
      {
        std::cout << "status after analysis 1: " << status_rocsparse_ << "\n";
      }

      status_rocsparse_ = rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(),
                                                    rocsparse_operation_none,
                                                    n,
                                                    U_csr_->getNnz(),
                                                    descr_U_,
                                                    U_csr_->getValues(ReSolve::memory::DEVICE), // vals_,
                                                    U_csr_->getRowData(ReSolve::memory::DEVICE),
                                                    U_csr_->getColData(ReSolve::memory::DEVICE),
                                                    info_U_,
                                                    rocsparse_analysis_policy_force,
                                                    rocsparse_solve_policy_auto,
                                                    U_buffer_);
      error_sum += status_rocsparse_;
      if (status_rocsparse_ != 0)
      {
        out::error() << "status after analysis 2: " << status_rocsparse_ << "\n";
      }

      // allocate aux data
      if (d_aux1_ == nullptr)
      {
        mem_.allocateArrayOnDevice(&d_aux1_, n);
      }
      if (d_aux2_ == nullptr)
      {
        mem_.allocateArrayOnDevice(&d_aux2_, n);
      }
    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  /**
   * @brief Refactorize the matrix A
   *
   * @post M_ is split into L and U factors
   */
  int LinSolverDirectRocSolverRf::refactorize()
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    int error_sum = 0;
    mem_.deviceSynchronize();
    status_rocblas_ = rocsolver_dcsrrf_refactlu(workspace_->getRocblasHandle(),
                                                A_->getNumRows(),
                                                A_->getNnz(),
                                                A_->getRowData(ReSolve::memory::DEVICE), // kRowPtr_,
                                                A_->getColData(ReSolve::memory::DEVICE), // jCol_,
                                                A_->getValues(ReSolve::memory::DEVICE),  // vals_,
                                                M_->getNnz(),
                                                M_->getRowData(ReSolve::memory::DEVICE),
                                                M_->getColData(ReSolve::memory::DEVICE),
                                                M_->getValues(ReSolve::memory::DEVICE), // OUTPUT,
                                                d_P_,
                                                d_Q_,
                                                infoM_);
    mem_.deviceSynchronize();
    error_sum += status_rocblas_;

    if (solve_mode_ == 1)
    {
      // split M, fill L and U with correct values
      status_rocblas_ = rocsolver_dcsrrf_splitlu(workspace_->getRocblasHandle(),
                                                 A_->getNumRows(),
                                                 M_->getNnz(),
                                                 M_->getRowData(ReSolve::memory::DEVICE),
                                                 M_->getColData(ReSolve::memory::DEVICE),
                                                 M_->getValues(ReSolve::memory::DEVICE), // vals_,
                                                 L_csr_->getRowData(ReSolve::memory::DEVICE),
                                                 L_csr_->getColData(ReSolve::memory::DEVICE),
                                                 L_csr_->getValues(ReSolve::memory::DEVICE), // vals_,
                                                 U_csr_->getRowData(ReSolve::memory::DEVICE),
                                                 U_csr_->getColData(ReSolve::memory::DEVICE),
                                                 U_csr_->getValues(ReSolve::memory::DEVICE));
      mem_.deviceSynchronize();
      error_sum += status_rocblas_;
    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  /**
   * @brief Solve the system of equations A*x = rhs
   *
   * @param[in,out] rhs - vector_type* - right-hand side
   *
   * @return int - sum of errors from solve
   */
  int LinSolverDirectRocSolverRf::solve(vector_type* rhs)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    int error_sum = 0;
    if (solve_mode_ == 0)
    {
      mem_.deviceSynchronize();
      status_rocblas_ = rocsolver_dcsrrf_solve(workspace_->getRocblasHandle(),
                                               A_->getNumRows(),
                                               1,
                                               M_->getNnz(),
                                               M_->getRowData(ReSolve::memory::DEVICE),
                                               M_->getColData(ReSolve::memory::DEVICE),
                                               M_->getValues(ReSolve::memory::DEVICE),
                                               d_P_,
                                               d_Q_,
                                               rhs->getData(ReSolve::memory::DEVICE),
                                               A_->getNumRows(),
                                               infoM_);
      mem_.deviceSynchronize();
    }
    else
    {
      // not implemented yet
      hip::permuteVectorP(A_->getNumRows(), d_P_, rhs->getData(ReSolve::memory::DEVICE), d_aux1_);
      mem_.deviceSynchronize();
      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             L_csr_->getNnz(),
                             &(constants::ONE),
                             descr_L_,
                             L_csr_->getValues(ReSolve::memory::DEVICE),
                             L_csr_->getRowData(ReSolve::memory::DEVICE),
                             L_csr_->getColData(ReSolve::memory::DEVICE),
                             info_L_,
                             d_aux1_,
                             d_aux2_, // result
                             rocsparse_solve_policy_auto,
                             L_buffer_);
      error_sum += status_rocsparse_;

      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             U_csr_->getNnz(),
                             &(constants::ONE),
                             descr_U_,
                             U_csr_->getValues(ReSolve::memory::DEVICE),
                             U_csr_->getRowData(ReSolve::memory::DEVICE),
                             U_csr_->getColData(ReSolve::memory::DEVICE),
                             info_U_,
                             d_aux2_, // input
                             d_aux1_, // result
                             rocsparse_solve_policy_auto,
                             U_buffer_);
      error_sum += status_rocsparse_;

      hip::permuteVectorQ(A_->getNumRows(), d_Q_, d_aux1_, rhs->getData(ReSolve::memory::DEVICE));
      mem_.deviceSynchronize();
    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  /**
   * @brief Solve the system of equations A*x = rhs
   *
   * @param[in] rhs - vector_type* - right-hand side
   * @param[out] x - vector_type* - solution
   *
   * @return int - sum of errors from solve
   */
  int LinSolverDirectRocSolverRf::solve(vector_type* rhs, vector_type* x)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    x->copyDataFrom(rhs->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::DEVICE);
    x->setDataUpdated(ReSolve::memory::DEVICE);
    int error_sum = 0;
    if (solve_mode_ == 0)
    {
      mem_.deviceSynchronize();
      status_rocblas_ = rocsolver_dcsrrf_solve(workspace_->getRocblasHandle(),
                                               A_->getNumRows(),
                                               1,
                                               M_->getNnz(),
                                               M_->getRowData(ReSolve::memory::DEVICE),
                                               M_->getColData(ReSolve::memory::DEVICE),
                                               M_->getValues(ReSolve::memory::DEVICE),
                                               d_P_,
                                               d_Q_,
                                               x->getData(ReSolve::memory::DEVICE),
                                               A_->getNumRows(),
                                               infoM_);
      error_sum += status_rocblas_;
      mem_.deviceSynchronize();
    }
    else
    {
      // not implemented yet

      hip::permuteVectorP(A_->getNumRows(), d_P_, rhs->getData(ReSolve::memory::DEVICE), d_aux1_);
      mem_.deviceSynchronize();

      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             L_csr_->getNnz(),
                             &(constants::ONE),
                             descr_L_,
                             L_csr_->getValues(ReSolve::memory::DEVICE), // vals_,
                             L_csr_->getRowData(ReSolve::memory::DEVICE),
                             L_csr_->getColData(ReSolve::memory::DEVICE),
                             info_L_,
                             d_aux1_,
                             d_aux2_, // result
                             rocsparse_solve_policy_auto,
                             L_buffer_);
      error_sum += status_rocsparse_;

      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             U_csr_->getNnz(),
                             &(constants::ONE),
                             descr_U_,
                             U_csr_->getValues(ReSolve::memory::DEVICE), // vals_,
                             U_csr_->getRowData(ReSolve::memory::DEVICE),
                             U_csr_->getColData(ReSolve::memory::DEVICE),
                             info_U_,
                             d_aux2_, // input
                             d_aux1_, // result
                             rocsparse_solve_policy_auto,
                             U_buffer_);
      error_sum += status_rocsparse_;

      hip::permuteVectorQ(A_->getNumRows(), d_Q_, d_aux1_, x->getData(ReSolve::memory::DEVICE));
      mem_.deviceSynchronize();
    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  /**
   * @brief Set the solve mode for LinSolverDirectRocSolverRf
   *
   * @param[in] mode - int - solve mode
   *
   * @return int - 0 if successful
   */
  int LinSolverDirectRocSolverRf::setSolveMode(int mode)
  {
    solve_mode_ = mode;
    return 0;
  }

  /**
   * @brief Get the solve mode for LinSolverDirectRocSolverRf
   *
   * @return int - solve mode
   */
  int LinSolverDirectRocSolverRf::getSolveMode() const
  {
    return solve_mode_;
  }

  /**
   * @brief Set the CLI parameters for LinSolverDirectRocSolverRf
   *
   * Currently only supports setting the solve mode
   * (rocsparse trisolve or default).
   *
   * @param[in] id - string - parameter ID
   * @param[in] value - string - parameter value
   */
  int LinSolverDirectRocSolverRf::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
    case SOLVE_MODE:
      if (value == "rocsparse_trisolve")
      {
        // use rocsparse triangular solver
        setSolveMode(1);
      }
      else
      {
        // use default
        setSolveMode(0);
      }
      break;
    default:
      std::cout << "Setting parameter failed!\n";
    }
    return 0;
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
  std::string LinSolverDirectRocSolverRf::getCliParamString(const std::string id) const
  {
    std::string value("");
    switch (getParamId(id))
    {
    case SOLVE_MODE:
      switch (getSolveMode())
      {
      case 0:
        value = "default";
        break;
      case 1:
        value = "rocsparse_trisolve";
        break;
      }
    default:
      out::error() << "Trying to get unknown string parameter " << id << "\n";
    }
    return value;
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
  index_type LinSolverDirectRocSolverRf::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown integer parameter " << id << "\n";
    }
    return -1;
  }

  /**
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return real_type Value of the real_type parameter to return.
   */
  real_type LinSolverDirectRocSolverRf::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown real parameter " << id << "\n";
    }
    return std::numeric_limits<real_type>::quiet_NaN();
  }

  /**
   * @brief Placeholder function for now.
   *
   * The following switch (getParamId(Id)) cases always run the default and
   * are currently redundant code (like an if (true)).
   * In the future, they will be expanded to include more options.
   *
   * @param id - string ID for parameter to get.
   * @return bool Value of the bool parameter to return.
   */
  bool LinSolverDirectRocSolverRf::getCliParamBool(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    }
    return false;
  }

  /**
   * @brief Placeholder function that shouldn't be called.
   */
  int LinSolverDirectRocSolverRf::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to print unknown parameter " << id << "\n";
      return 1;
    }
    return 0;
  }

  //
  // Private methods
  //

  /**
   * @brief Combine L and U factors already in CSR into a single matrix M
   *
   * M = [L U], where L and U are lower and upper triangular factors
   * The implicit identity diagonal of L is not included in M
   *
   * @param[in] L - matrix::Sparse* - lower triangular factor
   * @param[in] U - matrix::Sparse* - upper triangular factor
   *
   * @post M_ is allocated and filled with L and U factors
   */
  void LinSolverDirectRocSolverRf::combineFactors(matrix::Sparse* L, matrix::Sparse* U)
  {
    index_type  n     = L->getNumRows();
    index_type* L_row = L->getRowData(memory::HOST);
    index_type* L_col = L->getColData(memory::HOST);
    index_type* U_row = U->getRowData(memory::HOST);
    index_type* U_col = U->getColData(memory::HOST);
    index_type  M_nnz = (L->getNnz() + U->getNnz() - n);
    M_                = new matrix::Csr(n, n, M_nnz);
    M_->allocateMatrixData(memory::HOST);
    index_type* M_row = M_->getRowData(memory::HOST);
    index_type* M_col = M_->getColData(memory::HOST);
    // The total number of non-zeros in a row is the sum of non-zeros in L and U,
    // minus 1 to not count the diagonal element twice.
    // You can verify with this formula that M_row[i+1] - M_row[i] is the number of non-zeros in row i.
    // M_row[i+1] - M_row[i] = (L_row[i+1] - L_row[i]) + (U_row[i+1] - U_row[i]) - 1
    // The number of zeros in the i-th row of L + U -1.
    for (index_type i = 0; i <= n; i++)
    {
      M_row[i] = L_row[i] + U_row[i] - i;
    }
    // Now we need to fill the M_col array with the correct column indices.
    index_type count = 0;
    for (index_type i = 0; i < n; ++i)
    {
      for (index_type j = L_row[i]; j < L_row[i + 1]; ++j)
      {
        M_col[count++] = L_col[j];
      }
      for (index_type j = U_row[i] + 1; j < U_row[i + 1]; ++j) // skip the diagonal element of U, which is at U_row[i]
      {
        M_col[count++] = U_col[j];
      }
    }
  }

  /**
   * @brief initialize the parameter list for LinSolverDirectRocSolverRf
   *
   * currently only "solve_mode" is supported
   */
  void LinSolverDirectRocSolverRf::initParamList()
  {
    params_list_["solve_mode"] = SOLVE_MODE;
  }
} // namespace ReSolve
