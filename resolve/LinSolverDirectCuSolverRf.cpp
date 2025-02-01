#include <cassert>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
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
                                               L->getNnz(),
                                               L->getRowData(memory::DEVICE),
                                               L->getColData(memory::DEVICE),
                                               L->getValues( memory::DEVICE),
                                               U->getNnz(),
                                               U->getRowData(memory::DEVICE),
                                               U->getColData(memory::DEVICE),
                                               U->getValues( memory::DEVICE),
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

} // namespace resolve
