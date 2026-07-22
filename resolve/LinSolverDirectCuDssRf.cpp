#include "LinSolverDirectCuDssRf.hpp"

#include <cassert>
#include <cstring>

#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief Constructor for LinSolverDirectCuDssRf
   *
   * @param workspace - pointer to the LinAlgWorkspaceCUDA object (not used)
   */
  LinSolverDirectCuDssRf::LinSolverDirectCuDssRf(LinAlgWorkspaceCUDA* /* workspace */)
  {
    cudssCreate(&handle_cudss_);
    cudssConfigCreate(&config_cudss_);
    cudssDataCreate(handle_cudss_, &data_cudss_);
    setup_completed_ = false;
    initParamList();
  }

  /**
   * @brief Destructor for LinSolverDirectCuDssRf
   *
   * Destroys the cuDssRf handle, config, and data, then deletes the permutation vectors
   * from the device memory.
   *
   * @pre The cuDssRf handle, config, and data have been created.
   * @post The cuDssRf handle, config, and data have been destroyed.
   *
   * @pre The permutation vectors are allocated on the device.
   * @post The permutation vectors are deleted from the device.
   *
   * @note The permutation vectors are not deleted from the host memory.
   */
  LinSolverDirectCuDssRf::~LinSolverDirectCuDssRf()
  {
    cudssMatrixDestroy(descr_A_);
    cudssDataDestroy(handle_cudss_, data_cudss_);
    cudssConfigDestroy(config_cudss_);
    cudssDestroy(handle_cudss_);

    mem_.deleteOnDevice(d_P_);
  }

  /**
   * @brief For backward compatibility. Only A and P are needed.
   */
  int LinSolverDirectCuDssRf::setup(matrix::Sparse* A,
                                    matrix::Sparse*,
                                    matrix::Sparse*,
                                    index_type* P,
                                    index_type*,
                                    vector_type*)
  {
    return setup(A, P);
  }

  /**
   * @brief Setup the cuDssRf factorization with factors already in CSR
   *
   * Sets up the cuDssRf factorization for the given matrix A. The
   * permutation vector P is also set up.
   *
   * @param[in] A - pointer to the matrix A
   * @param[in] P - pointer to the permutation vector P
   *
   * @pre The matrix A is in CSR format.
   * @post Device storage for L and U is allocated and synchronized.
   */
  int LinSolverDirectCuDssRf::setup(matrix::Sparse* A,
                                    index_type*     P)
  {
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix A has to be in CSR format for cuDssRf input.\n");
    int error_sum  = 0;
    this->A_       = A;
    index_type n   = A_->getNumRows();
    index_type nnz = A_->getNnz();

    if (setup_completed_)
    {
      cudssMatrixDestroy(descr_A_);
      cudssDataDestroy(handle_cudss_, data_cudss_);
      cudssConfigDestroy(config_cudss_);

      descr_A_ = nullptr;
      cudssConfigCreate(&config_cudss_);
      cudssDataCreate(handle_cudss_, &data_cudss_);
    }

    if (d_P_ == nullptr)
    {
      mem_.allocateArrayOnDevice(&d_P_, n);
    }

    mem_.copyArrayHostToDevice(d_P_, P, n);

    if (d_P_ != nullptr)
    {
      cudssAlgType_t reorder_alg = CUDSS_ALG_1;
      cudssConfigSet(config_cudss_, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(cudssAlgType_t));
      cudssDataSet(handle_cudss_, data_cudss_, CUDSS_DATA_USER_PERM, d_P_, n * sizeof(index_type));
    }

    cudssStatus_t status = cudssMatrixCreateCsr(&descr_A_,
                                                n,
                                                n,
                                                nnz,
                                                A_->getRowData(memory::DEVICE),
                                                nullptr,
                                                A_->getColData(memory::DEVICE),
                                                A_->getValues(memory::DEVICE),
                                                CUDA_R_32I,
                                                CUDA_R_64F,
                                                CUDSS_MTYPE_GENERAL,
                                                CUDSS_MVIEW_FULL,
                                                CUDSS_BASE_ZERO);
    error_sum += status;

    status = cudssExecute(handle_cudss_, CUDSS_PHASE_ANALYSIS, config_cudss_, data_cudss_, descr_A_, nullptr, nullptr);
    error_sum += status;

    setup_completed_ = true;
    return error_sum;
  }

  /**
   * @brief Refactorizes the matrix A
   *
   * Refactorizes the matrix A using the cuDssRf handle.
   *
   * @pre The cuDssRf handle has been created.
   * @pre The matrix A is in CSR format.
   * @pre The permutation vectors P is allocated on the device.
   * @pre Matrix A's data is on the device.
   *
   * @post The matrix A is refactorized.
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuDssRf::refactorize()
  {
    assert(A_ != nullptr && "Matrix A is null!");
    assert(A_->getNumRows() > 0 && "Matrix A must have positive row count!");
    assert(A_->getNnz() > 0 && "Matrix A must have positive number of nonzeros!");

    return cudssExecute(handle_cudss_,
                        CUDSS_PHASE_FACTORIZATION,
                        config_cudss_,
                        data_cudss_,
                        descr_A_,
                        nullptr,
                        nullptr);
  }

  /**
   * @brief Solves the system of equations Ax=rhs
   *
   * Solves the system of equations Ax=rhs using the cuDssRf handle.
   * The solution overwrites the right-hand side vector.
   *
   * @param[in,out] rhs - pointer to right-hand side vector, changes to solution
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuDssRf::solve(vector_type* rhs)
  {
    return solve(rhs, rhs);
  }

  /**
   * @brief solves the system of equations Ax=rhs
   *
   * Solves the system of equations Ax=rhs using the cuDssRf handle.
   * The solution is stored in x.
   *
   * @param[in] rhs - pointer to right-hand side vector
   * @param[out] x - pointer to solution vector
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuDssRf::solve(vector_type* rhs, vector_type* x)
  {
    index_type n = A_->getNumRows();

    int error_sum = 0;

    cudssMatrix_t rhs_descr;
    cudssMatrix_t x_descr;
    error_sum += cudssMatrixCreateDn(&rhs_descr, n, 1, n, rhs->getData(memory::DEVICE), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    error_sum += cudssMatrixCreateDn(&x_descr, n, 1, n, x->getData(memory::DEVICE), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    error_sum += cudssExecute(handle_cudss_, CUDSS_PHASE_SOLVE, config_cudss_, data_cudss_, descr_A_, x_descr, rhs_descr);

    x->setDataUpdated(memory::DEVICE);

    cudssMatrixDestroy(rhs_descr);
    cudssMatrixDestroy(x_descr);

    return error_sum;
  }

  // For backward compatibility
  int LinSolverDirectCuDssRf::setNumericalProperties(real_type nzero, real_type)
  {
    return setNumericalProperties(nzero);
  }

  /**
   * @brief Sets a flag threshold for zero pivots and a boost factor
   *
   * Sets the zero flagging threshold and boost factor for the cuDssRf handle.
   * Must be called before setup()
   *
   * @param[in] nzero - zero flagging threshold
   * @param[in] nboost - boost factor ()
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuDssRf::setNumericalProperties(real_type nzero)
  {
    zero_pivot_ = nzero;
    return cudssConfigSet(config_cudss_,
                          CUDSS_CONFIG_PIVOT_EPSILON,
                          &zero_pivot_,
                          sizeof(real_type));
  }

  /**
   * @brief Sets the paramters from Cli to the cuDssRf handle
   *
   * @param[in] id - string ID for the parameter to set
   * @param[in] value - string value for the parameter to set
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuDssRf::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
    case ZERO_PIVOT:
      zero_pivot_ = atof(value.c_str());
      setNumericalProperties(zero_pivot_);
      break;
    case PIVOT_BOOST:
      out::warning() << "Pivot boost is not implemented for cuDSS refactor.\n";
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
  std::string LinSolverDirectCuDssRf::getCliParamString(const std::string id) const
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
  index_type LinSolverDirectCuDssRf::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown integer parameter " << id << "\n";
    }
    return -1;
  }

  real_type LinSolverDirectCuDssRf::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
    case ZERO_PIVOT:
      return zero_pivot_;
    case PIVOT_BOOST:
      out::warning() << "Pivot boost is not implemented for cuDSS refactor.\n";
      break;
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
  bool LinSolverDirectCuDssRf::getCliParamBool(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    }
    return false;
  }

  /**
   * @brief Prints the parameters from Cli to the console
   */
  int LinSolverDirectCuDssRf::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    case ZERO_PIVOT:
      std::cout << zero_pivot_ << "\n";
      break;
    case PIVOT_BOOST:
      out::warning() << "Pivot boost is not implemented for cuDSS refactor.\n";
      break;
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
   * @brief Set the zero pivot and pivot boost parameters
   */
  void LinSolverDirectCuDssRf::initParamList()
  {
    params_list_["zero_pivot"]  = ZERO_PIVOT;
    params_list_["pivot_boost"] = PIVOT_BOOST;
  }

  /**
   * @brief Convert CSC to CSR matrix on the host
   *
   * @authors Slaven Peles <peless@ornl.gov>, Daniel Reynolds (SMU), and
   * David Gardner and Carol Woodward (LLNL)
   *
   * @param[in] A_csc - pointer to the CSC matrix
   * @param[out] A_csr - pointer to an empty CSR matrix
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectCuDssRf::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    // int error_sum = 0; TODO: Collect error output!
    assert(A_csc->getNnz() == A_csr->getNnz());
    assert(A_csc->getNumRows() == A_csr->getNumRows());
    assert(A_csr->getNumColumns() == A_csc->getNumColumns());

    A_csr->allocateMatrixData(memory::HOST);

    index_type nnz = A_csc->getNnz();
    index_type n   = A_csc->getNumColumns();

    index_type* rowIdxCsc = A_csc->getRowData(memory::HOST);
    index_type* colPtrCsc = A_csc->getColData(memory::HOST);
    real_type*  valuesCsc = A_csc->getValues(memory::HOST);

    index_type* rowPtrCsr = A_csr->getRowData(memory::HOST);
    index_type* colIdxCsr = A_csr->getColData(memory::HOST);
    real_type*  valuesCsr = A_csr->getValues(memory::HOST);

    // Set all CSR row pointers to zero
    for (index_type i = 0; i <= n; ++i)
    {
      rowPtrCsr[i] = 0;
    }

    // Set all CSR values and column indices to zero
    for (index_type i = 0; i < nnz; ++i)
    {
      colIdxCsr[i] = 0;
      valuesCsr[i] = 0.0;
    }

    // Compute number of entries per row
    for (index_type i = 0; i < nnz; ++i)
    {
      rowPtrCsr[rowIdxCsc[i]]++;
    }

    // Compute cumualtive sum of nnz per row
    for (index_type row = 0, rowsum = 0; row < n; ++row)
    {
      // Store value in row pointer to temp
      index_type temp = rowPtrCsr[row];

      // Copy cumulative sum to the row pointer
      rowPtrCsr[row] = rowsum;

      // Update row sum
      rowsum += temp;
    }
    rowPtrCsr[n] = nnz;

    for (index_type col = 0; col < n; ++col)
    {
      // Compute positions of column indices and values in CSR matrix and store them there
      // Overwrites CSR row pointers in the process
      // adding to them the number of elements in that row
      for (index_type jj = colPtrCsc[col]; jj < colPtrCsc[col + 1]; jj++)
      {
        index_type row  = rowIdxCsc[jj];
        index_type dest = rowPtrCsr[row];

        colIdxCsr[dest] = col;
        valuesCsr[dest] = valuesCsc[jj];

        rowPtrCsr[row]++;
      }
    }

    // Restore CSR row pointer values
    // All values in rowPtrCsr have shifted by the number of elements in that row
    // for i>=1: new rowPtrCsr[i] = old rowPtrCsr[i-1] and new rowPtrCsr[0]=0
    for (index_type row = 0, last = 0; row <= n; row++)
    {
      index_type temp = rowPtrCsr[row];
      rowPtrCsr[row]  = last;
      last            = temp;
    }

    // Mark data on the host as updated
    A_csr->setUpdated(memory::HOST);

    return 0;
  }

} // namespace ReSolve
