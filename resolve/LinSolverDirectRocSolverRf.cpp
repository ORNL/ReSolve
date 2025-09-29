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
   */
  LinSolverDirectRocSolverRf::LinSolverDirectRocSolverRf(LinAlgWorkspaceHIP* workspace)
  {
    workspace_  = workspace;
    infoM_      = nullptr;
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
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  /**
   * @brief Set the CLI parameters for LinSolverDirectRocSolverRf
   *
   * Placeholder function till command line parameters are supported.
   *
   * @param[in] id - string - parameter ID
   * @param[in] value - string - parameter value
   */
  int LinSolverDirectRocSolverRf::setCliParam(const std::string id, const std::string value)
  {
    // Suppress unused variable warnings.
    (void) id;
    (void) value;
    return 0;
  }

  /**
   * @brief Placeholder function till command line parameters are supported.
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
    // Suppress unused variable warnings.
    (void) id;
    std::string value("");
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
   * currently no parameters are supported
   */
  void LinSolverDirectRocSolverRf::initParamList()
  {
  }
} // namespace ReSolve
