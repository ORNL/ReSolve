#include "LinSolverDirectKLU.hpp"

#include <cassert>
#include <cstring>

#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief Constructor for LinSolverDirectKLU
   *
   * Initializes the KLU solver with default parameters.
   */
  LinSolverDirectKLU::LinSolverDirectKLU()
  {
    Symbolic_ = nullptr;
    Numeric_  = nullptr;

    L_ = nullptr;
    U_ = nullptr;

    // Populate KLU data structure holding solver parameters
    klu_defaults(&Common_);
    Common_.btf              = 0;
    Common_.scale            = -1;
    Common_.ordering         = ordering_;
    Common_.tol              = pivot_threshold_tol_;
    Common_.halt_if_singular = halt_if_singular_;

    // Register configurable parameters
    initParamList();

    out::summary() << "KLU solver set with parameters:\n"
                   << "\tbtf              = " << Common_.btf << "\n"
                   << "\tscale            = " << Common_.scale << "\n"
                   << "\tordering         = " << Common_.ordering << "\n"
                   << "\tpivot threshold  = " << Common_.tol << "\n"
                   << "\thalt if singular = " << Common_.halt_if_singular << "\n";
  }

  /**
   * @brief Destructor for LinSolverDirectKLU
   *
   * Frees memory allocated for the KLU solver.
   * Deletes factors L_, U_, and permutation vectors P_ and Q_
   * if factors have been extracted.
   * Frees memory allocated for Symbolic_ and Numeric_.
   * @post All memory allocated for KLU solver is freed.
   * @post All pointers are set to nullptr.
   */
  LinSolverDirectKLU::~LinSolverDirectKLU()
  {
    if (factors_extracted_)
    {
      delete L_;
      delete U_;
      delete[] P_;
      delete[] Q_;
      L_ = nullptr;
      U_ = nullptr;
      P_ = nullptr;
      Q_ = nullptr;
    }
    klu_free_symbolic(&Symbolic_, &Common_);
    klu_free_numeric(&Numeric_, &Common_);
  }

  /**
   * @brief Setup the KLU solver with the matrix A.
   *
   * @param[in] A   - matrix to solve
   * @param[in] L   - optional lower triangular factor
   * @param[in] U   - optional upper triangular factor
   * @param[in] P   - optional row permutation vector
   * @param[in] Q   - optional column permutation vector
   * @param[in] rhs - optional right-hand side vector
   */
  int LinSolverDirectKLU::setup(matrix::Sparse* A,
                                matrix::Sparse* /* L */,
                                matrix::Sparse* /* U */,
                                index_type* /* P */,
                                index_type* /* Q */,
                                vector_type* /* rhs */)
  {
    this->A_ = A;
    return 0;
  }

  /**
   * @brief Analyze the matrix A and compute symbolic factorization.
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectKLU::analyze()
  {
    // in case we called this function AGAIN
    if (Symbolic_ != nullptr)
    {
      klu_free_symbolic(&Symbolic_, &Common_);
    }
    Symbolic_          = klu_analyze(A_->getNumRows(),
                            A_->getRowData(memory::HOST),
                            A_->getColData(memory::HOST),
                            &Common_);
    factors_extracted_ = false;

    if (L_ != nullptr)
    {
      delete L_;
      L_ = nullptr;
    }

    if (U_ != nullptr)
    {
      delete U_;
      U_ = nullptr;
    }

    if (Symbolic_ == nullptr)
    {
      out::error() << "Symbolic_ factorization failed with Common_.status = "
                   << Common_.status << "\n";
      return 1;
    }
    return 0;
  }

  /**
   * @brief Factorize the matrix A.
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectKLU::factorize()
  {
    if (Numeric_ != nullptr)
    {
      klu_free_numeric(&Numeric_, &Common_);
    }

    Numeric_ = klu_factor(A_->getRowData(memory::HOST),
                          A_->getColData(memory::HOST),
                          A_->getValues(memory::HOST),
                          Symbolic_,
                          &Common_);

    factors_extracted_ = false;

    if (L_ != nullptr)
    {
      delete L_;
      L_ = nullptr;
    }

    if (U_ != nullptr)
    {
      delete U_;
      U_ = nullptr;
    }

    if (Numeric_ == nullptr)
    {
      return 1;
    }
    else
    {
      if (Numeric_->nzoff != 0)
      {
        assert(0 && "Numeric_->nzoff != 0, this is not supported by ReSolve!");
      }
    }
    return 0;
  }

  /**
   * @brief Update the factorization of the matrix A.
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectKLU::refactorize()
  {
    int kluStatus = klu_refactor(A_->getRowData(memory::HOST),
                                 A_->getColData(memory::HOST),
                                 A_->getValues(memory::HOST),
                                 Symbolic_,
                                 Numeric_,
                                 &Common_);

    factors_extracted_ = false;

    if (L_ != nullptr)
    {
      delete L_;
      L_ = nullptr;
    }

    if (U_ != nullptr)
    {
      delete U_;
      U_ = nullptr;
    }

    if (!kluStatus)
    {
      // display error
      return 1;
    }
    return 0;
  }

  /**
   * @brief Solves a system of equations A*x = rhs.
   *
   * @param[in] rhs - right-hand side vector
   * @param[out] x   - solution vector
   *
   * @return 0 if successful, 1 otherwise
   */

  int LinSolverDirectKLU::solve(vector_type* rhs, vector_type* x)
  {
    // copy the vector
    x->copyDataFrom(rhs->getData(memory::HOST), memory::HOST, memory::HOST);
    x->setDataUpdated(memory::HOST);
    int kluStatus = 1;
    // check sparsity format of A
    if (A_->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_COLUMN)
    {
      kluStatus = klu_solve(Symbolic_, Numeric_, A_->getNumRows(), 1, x->getData(memory::HOST), &Common_);
    }
    else if (A_->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW)
    {
      kluStatus = klu_tsolve(Symbolic_, Numeric_, A_->getNumRows(), 1, x->getData(memory::HOST), &Common_);
    }
    else
    {
      out::error() << "Unsupported sparse format for matrix A in LinSolverDirectKLU! Only CSC and CSR are supported.\n";
      return 1;
    }

    if (!kluStatus)
    {
      return 1;
    }
    return 0;
  }

  /**
   * @brief Generic solver with matrix A with unspecified rhs (not implemented).
   */
  int LinSolverDirectKLU::solve(vector_type*)
  {
    out::error() << "Function solve(Vector* x) not implemented in LinSolverDirectKLU!\n"
                 << "Consider using solve(Vector* rhs, Vector* x) instead.\n";
    return 1;
  }

  /**
   * @brief Extracts L and U factors from the KLU solver in CSR format, if they have not already been extracted.
   *
   * It extracts the factors as $A = U^T L^T$,
   * where U^T is the reinterpretation of the CSC U factor as CSR and L^T is the reinterpretation of the CSC L factor as CSR.
   */
  void LinSolverDirectKLU::extractFactorsCsr()
  {
    if (!factors_extracted_)
    {
      const int nnzL = Numeric_->lnz;
      const int nnzU = Numeric_->unz;

      // Create CSR matrices - L gets U's data, U gets L's data
      L_ = new matrix::Csr(A_->getNumRows(), A_->getNumColumns(), nnzU);
      U_ = new matrix::Csr(A_->getNumRows(), A_->getNumColumns(), nnzL);
      L_->allocateMatrixData(memory::HOST);
      U_->allocateMatrixData(memory::HOST);

      int ok = klu_extract(Numeric_,
                           Symbolic_,
                           U_->getRowData(memory::HOST), // L CSC colptr -> U CSR rowptr
                           U_->getColData(memory::HOST), // L CSC rowidx -> U CSR colidx
                           U_->getValues(memory::HOST),  // L CSC values -> U CSR values
                           L_->getRowData(memory::HOST), // U CSC colptr -> L CSR rowptr
                           L_->getColData(memory::HOST), // U CSC rowidx -> L CSR colidx
                           L_->getValues(memory::HOST),  // U CSC values -> L CSR values
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           &Common_);

      L_->setUpdated(memory::HOST);
      U_->setUpdated(memory::HOST);
      (void) ok; // TODO: Check status in ok before setting `factors_extracted_`
      factors_extracted_ = true;
    }
    return;
  }

  /**
   * @brief Gets an L factor of the matrix A in compressed sparse row format.
   *
   * @return L factor in compressed sparse row format
   */
  matrix::Sparse* LinSolverDirectKLU::getLFactorCsr()
  {
    extractFactorsCsr();
    return L_;
  }

  /**
   * @brief Gets a U factor of the matrix A in compressed sparse row format.
   *
   * @return U factor in compressed sparse row format
   */
  matrix::Sparse* LinSolverDirectKLU::getUFactorCsr()
  {
    extractFactorsCsr();
    return U_;
  }

  /**
   * @brief Extract L and U factors from the KLU solver in compressed sparse column format.
   */
  void LinSolverDirectKLU::extractFactors()
  {
    if (!factors_extracted_)
    {
      const int nnzL = Numeric_->lnz;
      const int nnzU = Numeric_->unz;

      L_ = new matrix::Csc(A_->getNumRows(), A_->getNumColumns(), nnzL);
      U_ = new matrix::Csc(A_->getNumRows(), A_->getNumColumns(), nnzU);
      L_->allocateMatrixData(memory::HOST);
      U_->allocateMatrixData(memory::HOST);

      int ok = klu_extract(Numeric_,
                           Symbolic_,
                           L_->getColData(memory::HOST),
                           L_->getRowData(memory::HOST),
                           L_->getValues(memory::HOST),
                           U_->getColData(memory::HOST),
                           U_->getRowData(memory::HOST),
                           U_->getValues(memory::HOST),
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           &Common_);

      L_->setUpdated(memory::HOST);
      U_->setUpdated(memory::HOST);
      (void) ok; // TODO: Check status in ok before setting `factors_extracted_`
      factors_extracted_ = true;
    }
  }

  /**
   * @brief Get the lower triangular factor L of the matrix A in compressed sparse column format.
   *
   * @return L factor in compressed sparse column format
   */
  matrix::Sparse* LinSolverDirectKLU::getLFactor()
  {
    extractFactors();
    return L_;
  }

  /**
   * @brief Get the upper triangular factor U of the matrix A in compressed sparse column format.
   *
   * @return U factor
   */
  matrix::Sparse* LinSolverDirectKLU::getUFactor()
  {
    extractFactors();
    return U_;
  }

  /**
   * @brief Get the permutation vector P.
   *
   * @return P permutation vector
   */
  index_type* LinSolverDirectKLU::getPOrdering()
  {
    if (Numeric_ != nullptr)
    {
      P_           = new index_type[A_->getNumRows()];
      size_t nrows = static_cast<size_t>(A_->getNumRows());
      std::memcpy(P_, Symbolic_->P, nrows * sizeof(index_type));
      return P_;
    }
    else
    {
      return nullptr;
    }
  }

  /**
   * @brief Get the permutation vector Q.
   *
   * @return Q permutation vector
   */
  index_type* LinSolverDirectKLU::getQOrdering()
  {
    if (Numeric_ != nullptr)
    {
      Q_           = new index_type[A_->getNumRows()];
      size_t nrows = static_cast<size_t>(A_->getNumRows());
      std::memcpy(Q_, Symbolic_->Q, nrows * sizeof(index_type));
      return Q_;
    }
    else
    {
      return nullptr;
    }
  }

  /**
   * @brief Set the pivot threshold for the KLU solver.
   *
   * @param[in] tol - pivot threshold
   *
   * @post Common_.tol is set to tol
   */
  void LinSolverDirectKLU::setPivotThreshold(real_type tol)
  {
    pivot_threshold_tol_ = tol;
    Common_.tol          = tol;
  }

  /**
   * @brief Set the ordering for the KLU solver.
   *
   * @param[in] ordering - ordering method
   *
   * @post Common_.ordering is set to ordering
   */
  void LinSolverDirectKLU::setOrdering(int ordering)
  {
    ordering_        = ordering;
    Common_.ordering = ordering;
  }

  /**
   * @brief Set the halt if singular flag for the KLU solver.
   *
   * Tells the solver to halt if a singular matrix is detected.
   *
   * @param[in] isHalt - halt if singular flag
   */

  void LinSolverDirectKLU::setHaltIfSingular(bool isHalt)
  {
    halt_if_singular_        = isHalt;
    Common_.halt_if_singular = isHalt;
  }

  /**
   * @brief Get the condition number of the matrix A.
   *
   * @return condition number of the matrix A
   */
  real_type LinSolverDirectKLU::getMatrixConditionNumber()
  {
    klu_rcond(Symbolic_, Numeric_, &Common_);
    return Common_.rcond;
  }

  /**
   * @brief set Cli parameters for KLU solver.
   *
   * @param[in] id    - string ID for parameter to set
   * @param[in] value - string value for parameter to set
   *
   * @post KLU solver parameters id is set to value
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectKLU::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
    case PIVOT_TOL:
      setPivotThreshold(atof(value.c_str()));
      break;
    case ORDERING:
      setOrdering(atoi(value.c_str()));
      break;
    case HALT_IF_SINGULAR:
      setHaltIfSingular(value == "yes");
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
  std::string LinSolverDirectKLU::getCliParamString(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown string parameter " << id << "\n";
    }
    return "";
  }

  /**
   * @brief Get the integer parameter for the KLU solver.
   *
   * Currently, the only integer parameter is the ordering method.
   *
   * @param[in] id - string ID for parameter to get
   *
   * @return index_type - represents the ordering method
   */
  index_type LinSolverDirectKLU::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
    case ORDERING:
      return ordering_;
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
  real_type LinSolverDirectKLU::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
    case PIVOT_TOL:
      return pivot_threshold_tol_;
    default:
      out::error() << "Trying to get unknown real parameter " << id << "\n";
    }
    return std::numeric_limits<real_type>::quiet_NaN();
  }

  /**
   * @brief Get the boolean parameter for the KLU solver.
   *
   * Currently, the only boolean parameter is the halt if singular flag.
   *
   * @param[in] id - string ID for parameter to get
   *
   * @return bool - true if halt if singular flag is set to true, false otherwise
   */
  bool LinSolverDirectKLU::getCliParamBool(const std::string id) const
  {
    switch (getParamId(id))
    {
    case HALT_IF_SINGULAR:
      return halt_if_singular_;
    default:
      out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    }
    return false;
  }

  /**
   * @brief Print the KLU solver Cli parameters.
   *
   * @param[in] id - string ID for parameter to print
   *
   * @return 0 if successful, 1 otherwise
   */
  int LinSolverDirectKLU::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    case PIVOT_TOL:
      std::cout << pivot_threshold_tol_ << "\n";
      break;
    case ORDERING:
      std::cout << ordering_ << "\n";
      break;
    case HALT_IF_SINGULAR:
      std::cout << halt_if_singular_ << "\n";
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
   * @brief Initialize the parameter list for KLU solver.
   *
   * @post params_list_ is populated with the KLU solver parameters:
   * - pivot_tol
   * - ordering
   * - halt_if_singular
   */
  void LinSolverDirectKLU::initParamList()
  {
    params_list_["pivot_tol"]        = PIVOT_TOL;
    params_list_["ordering"]         = ORDERING;
    params_list_["halt_if_singular"] = HALT_IF_SINGULAR;
  }

} // namespace ReSolve
