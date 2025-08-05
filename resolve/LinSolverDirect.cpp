/**
 * @file LinSolverDirect.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Implementation of direct solver base class.
 *
 */
#include <resolve/LinSolverDirect.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve

{
  using out = io::Logger;
  using vector_type = vector::Vector;

  /**
   * @brief Constructor for LinSolverDirect class.
   *
   * Initializes pointers to L, U, P, and Q to nullptr.
   */
  LinSolverDirect::LinSolverDirect()
  {
    L_ = nullptr;
    U_ = nullptr;
    P_ = nullptr;
    Q_ = nullptr;
  }

  /**
   * @brief Empty destructor for LinSolverDirect class.
   */
  LinSolverDirect::~LinSolverDirect()
  {
  }

  /**
   * @brief Setup function for LinSolverDirect class.
   *
   * @param[in] A - matrix to be solved
   * @param[in] L - optional lower triangular factor
   * @param[in] U - optional upper triangular factor
   * @param[in] P - optional row permutation vector
   * @param[in] Q - optional column permutation vector
   * @param[in] rhs - optional right-hand side vector
   *
   * @return int - error code, 0 if successful
   */
  int LinSolverDirect::setup(matrix::Sparse* A,
                             matrix::Sparse* /* L */,
                             matrix::Sparse* /* U */,
                             index_type* /* P */,
                             index_type* /* Q */,
                             vector_type* /* rhs */)
  {
    if (A == nullptr)
    {
      return 1;
    }
    A_ = A;
    return 0;
  }

  /**
   * @brief Setup function for LinSolverDirect class with CSR data
   *
   * @param[in] A - matrix to be solved
   * @param[in] L - optional lower triangular factor
   * @param[in] U - optional upper triangular factor
   * @param[in] P - optional row permutation vector
   * @param[in] Q - optional column permutation vector
   * @param[in] rhs - optional right-hand side vector
   *
   * @return int - error code, 0 if successful
   */
  int LinSolverDirect::setupCsr(matrix::Sparse* A,
                                matrix::Sparse* /* L */,
                                matrix::Sparse* /* U */,
                                index_type* /* P */,
                                index_type* /* Q */,
                                vector_type* /* rhs */)
  {
    if (A == nullptr)
    {
      return 1;
    }
    A_ = A;
    return 0;
  }

  /**
   * @brief Placeholder function for symbolic factorization.
   */
  int LinSolverDirect::analyze()
  {
    return 1;
  }

  /**
   * @brief Placeholder function for numeric factorization.
   */
  int LinSolverDirect::factorize()
  {
    return 1;
  }

  /**
   * @brief Placeholder function for refactorization.
   */
  int LinSolverDirect::refactorize()
  {
    return 1;
  }

  /**
   * @brief Placeholder function for lower triangular factor in Csr.
   */
  matrix::Sparse* LinSolverDirect::getLFactorCsr()
  {
    return nullptr;
  }

  /**
   * @brief Placeholder function for upper triangular factor in Csr.
   */
  matrix::Sparse* LinSolverDirect::getUFactorCsr()
  {
    return nullptr;
  }

  /**
   * @brief Placeholder function for scaling R_ vector
   */
  vector_type* LinSolverDirect::getRFactorCsr()
  {
    std::cout << "getRFactorCsr called, but not implemented in LinSolverDirect.\n";
    return nullptr;
  }

  /**
   * @brief Placeholder function for lower triangular factor.
   */
  matrix::Sparse* LinSolverDirect::getLFactor()
  {
    return nullptr;
  }

  /**
   * @brief Placeholder function for upper triangular factor.
   */
  matrix::Sparse* LinSolverDirect::getUFactor()
  {
    return nullptr;
  }

  /**
   * @brief Placeholder function for row permutation vector.
   */
  index_type* LinSolverDirect::getPOrdering()
  {
    return nullptr;
  }

  /**
   * @brief Placeholder function for column permutation vector.
   */
  index_type* LinSolverDirect::getQOrdering()
  {
    return nullptr;
  }

} // namespace ReSolve
