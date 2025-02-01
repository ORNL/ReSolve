#include <cassert>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include "LinSolverDirectCuSolverRf.hpp"

namespace ReSolve 
{
  using out = io::Logger;

  LinSolverDirectCuSolverRf::LinSolverDirectCuSolverRf(LinAlgWorkspaceCUDA* /* workspace */)
  {
    cusolverRfCreate(&handle_cusolverrf_);
    setup_completed_ = false;
    initParamList();
  }

  LinSolverDirectCuSolverRf::~LinSolverDirectCuSolverRf()
  {
    cusolverRfDestroy(handle_cusolverrf_);
    mem_.deleteOnDevice(d_P_);
    mem_.deleteOnDevice(d_Q_);
    mem_.deleteOnDevice(d_T_);
  }

  int LinSolverDirectCuSolverRf::setup(matrix::Sparse* A,
                                       matrix::Sparse* L,
                                       matrix::Sparse* U,
                                       index_type* P,
                                       index_type* Q,
                                       vector_type* /* rhs */)
  {
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
           "Matrix A has to be in CSR format for cusolverRf input.\n");
    assert(L->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_COLUMN &&
           "Matrix L has to be in CSC format for cusolverRf input.\n");

    int error_sum = 0;
    this->A_ = A;
    index_type n = A_->getNumRows();

    //remember - P and Q are generally CPU variables
    // factorization data is stored in the handle. 
    // If function is called again, destroy the old handle to get rid of old data. 
    if (setup_completed_) {
      cusolverRfDestroy(handle_cusolverrf_);
      cusolverRfCreate(&handle_cusolverrf_);
    }

    auto* L_csc = static_cast<matrix::Csc*>(L);
    auto* U_csc = static_cast<matrix::Csc*>(U);
    auto* L_csr = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
    auto* U_csr = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
    csc2csr(L_csc, L_csr);
    csc2csr(U_csc, U_csr);
    L_csr->syncData(ReSolve::memory::DEVICE);
    U_csr->syncData(ReSolve::memory::DEVICE);

    if (d_P_ == nullptr){
      mem_.allocateArrayOnDevice(&d_P_, n);
    } 

    if (d_Q_ == nullptr){
      mem_.allocateArrayOnDevice(&d_Q_, n);
    }

    if (d_T_ != nullptr){
      mem_.deleteOnDevice(d_T_);
    }
    
    mem_.allocateArrayOnDevice(&d_T_, n);

    mem_.copyArrayHostToDevice(d_P_, P, n);
    mem_.copyArrayHostToDevice(d_Q_, Q, n);


    status_cusolverrf_ = cusolverRfSetResetValuesFastMode(handle_cusolverrf_, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);
    error_sum += status_cusolverrf_;
    status_cusolverrf_ = cusolverRfSetupDevice(n, 
                                               A_->getNnz(),
                                               A_->getRowData(memory::DEVICE),
                                               A_->getColData(memory::DEVICE),
                                               A_->getValues( memory::DEVICE),
                                               L_csr->getNnz(),
                                               L_csr->getRowData(memory::DEVICE),
                                               L_csr->getColData(memory::DEVICE),
                                               L_csr->getValues( memory::DEVICE),
                                               U_csr->getNnz(),
                                               U_csr->getRowData(memory::DEVICE),
                                               U_csr->getColData(memory::DEVICE),
                                               U_csr->getValues( memory::DEVICE),
                                               d_P_,
                                               d_Q_,
                                               handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    mem_.deviceSynchronize();
    status_cusolverrf_ = cusolverRfAnalyze(handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    const cusolverRfFactorization_t fact_alg =
      CUSOLVERRF_FACTORIZATION_ALG0;  // 0 - default, 1 or 2
    const cusolverRfTriangularSolve_t solve_alg =
      CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;  //  1- default, 2 or 3 // 1 causes error
    this->setAlgorithms(fact_alg, solve_alg);
    
    setup_completed_ = true;
    delete L_csr;
    delete U_csr;
    
    return error_sum;
  }

  void LinSolverDirectCuSolverRf::setAlgorithms(cusolverRfFactorization_t fact_alg,
                                                cusolverRfTriangularSolve_t solve_alg)
  {
    cusolverRfSetAlgs(handle_cusolverrf_, fact_alg, solve_alg);
  }

  int LinSolverDirectCuSolverRf::refactorize()
  {
    int error_sum = 0;
    status_cusolverrf_ = cusolverRfResetValues(A_->getNumRows(), 
                                               A_->getNnz(), 
                                               A_->getRowData(memory::DEVICE),
                                               A_->getColData(memory::DEVICE),
                                               A_->getValues( memory::DEVICE),
                                               d_P_,
                                               d_Q_,
                                               handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    mem_.deviceSynchronize();
    status_cusolverrf_ =  cusolverRfRefactor(handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    return error_sum; 
  }

  // solution is returned in RHS
  int LinSolverDirectCuSolverRf::solve(vector_type* rhs)
  {
    status_cusolverrf_ =  cusolverRfSolve(handle_cusolverrf_,
                                          d_P_,
                                          d_Q_,
                                          1,
                                          d_T_,
                                          A_->getNumRows(),
                                          rhs->getData(memory::DEVICE),
                                          A_->getNumRows());
    return status_cusolverrf_;
  }

  int LinSolverDirectCuSolverRf::solve(vector_type* rhs, vector_type* x)
  {
    x->copyDataFrom(rhs->getData(memory::DEVICE), memory::DEVICE, memory::DEVICE);
    x->setDataUpdated(memory::DEVICE);
    status_cusolverrf_ =  cusolverRfSolve(handle_cusolverrf_,
                                          d_P_,
                                          d_Q_,
                                          1,
                                          d_T_,
                                          A_->getNumRows(),
                                          x->getData(memory::DEVICE),
                                          A_->getNumRows());
    return status_cusolverrf_;
  }

  int LinSolverDirectCuSolverRf::setNumericalProperties(real_type nzero,
                                                        real_type nboost)
  {
    // Zero flagging threshold and boost NEED TO BE DOUBLE!
    double zero = static_cast<double>(nzero);
    double boost = static_cast<double>(nboost);
    status_cusolverrf_ = cusolverRfSetNumericProperties(handle_cusolverrf_,
                                                        zero,
                                                        boost);
    return status_cusolverrf_;
  }

  int LinSolverDirectCuSolverRf::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
      case ZERO_PIVOT:
        zero_pivot_ = atof(value.c_str());
        setNumericalProperties(zero_pivot_, pivot_boost_);
        break;
      case PIVOT_BOOST:
        pivot_boost_ = atof(value.c_str());
        setNumericalProperties(zero_pivot_, pivot_boost_);
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
  std::string LinSolverDirectCuSolverRf::getCliParamString(const std::string id) const
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
  index_type LinSolverDirectCuSolverRf::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
      default:
        out::error() << "Trying to get unknown integer parameter " << id << "\n";
    }
    return -1;
  }

  real_type LinSolverDirectCuSolverRf::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
      case ZERO_PIVOT:
        return zero_pivot_;
      case PIVOT_BOOST:
        return pivot_boost_;
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
  bool LinSolverDirectCuSolverRf::getCliParamBool(const std::string id) const
  {
    switch (getParamId(id))
    {
      default:
        out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    }
    return false;
  }

  int LinSolverDirectCuSolverRf::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
      case ZERO_PIVOT:
        std::cout << zero_pivot_ << "\n";
        break;
      case PIVOT_BOOST:
        std::cout << pivot_boost_ << "\n";
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

  void LinSolverDirectCuSolverRf::initParamList()
  {
    params_list_["zero_pivot"]  = ZERO_PIVOT;
    params_list_["pivot_boost"] = PIVOT_BOOST;
  }
  
  /**
   * @brief Convert CSC to CSR matrix on the host
   * 
   * @authors Slaven Peles <peless@ornl.gov>, Daniel Reynolds (SMU), and
   * David Gardner and Carol Woodward (LLNL)
   */
  int LinSolverDirectCuSolverRf::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    // int error_sum = 0; TODO: Collect error output!
    assert(A_csc->getNnz() == A_csr->getNnz());
    assert(A_csc->getNumRows() == A_csr->getNumColumns());
    assert(A_csr->getNumRows() == A_csc->getNumColumns());

    A_csr->allocateMatrixData(memory::HOST);

    index_type nnz = A_csc->getNnz();
    index_type n   = A_csc->getNumColumns();

    index_type* rowIdxCsc = A_csc->getRowData(memory::HOST);
    index_type* colPtrCsc = A_csc->getColData(memory::HOST);
    real_type*  valuesCsc = A_csc->getValues( memory::HOST);

    index_type* rowPtrCsr = A_csr->getRowData(memory::HOST);
    index_type* colIdxCsr = A_csr->getColData(memory::HOST);
    real_type*  valuesCsr = A_csr->getValues( memory::HOST);

    // Set all CSR row pointers to zero
    for (index_type i = 0; i <= n; ++i) {
      rowPtrCsr[i] = 0;
    }

    // Set all CSR values and column indices to zero
    for (index_type i = 0; i < nnz; ++i) {
      colIdxCsr[i] = 0;
      valuesCsr[i] = 0.0;
    }

    // Compute number of entries per row
    for (index_type i = 0; i < nnz; ++i) {
      rowPtrCsr[rowIdxCsc[i]]++;
    }

    // Compute cumualtive sum of nnz per row
    for (index_type row = 0, rowsum = 0; row < n; ++row)
    {
      // Store value in row pointer to temp
      index_type temp  = rowPtrCsr[row];

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
      for (index_type jj = colPtrCsc[col]; jj < colPtrCsc[col+1]; jj++)
      {
          index_type row  = rowIdxCsc[jj];
          index_type dest = rowPtrCsr[row];

          colIdxCsr[dest] = col;
          valuesCsr[dest] = valuesCsc[jj];

          rowPtrCsr[row]++;
      }
    }

    // Restore CSR row pointer values
    for (index_type row = 0, last = 0; row <= n; row++)
    {
        index_type temp  = rowPtrCsr[row];
        rowPtrCsr[row] = last;
        last    = temp;
    }

    // Mark data on the host as updated
    A_csr->setUpdated(memory::HOST);

    return 0;
  }

} // namespace resolve
