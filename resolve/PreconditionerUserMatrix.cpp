/**
 * @file   PreconditionerUserMatrix.cpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Implementation of preconditioner class with user matrix
 *
 */

#include "PreconditionerUserMatrix.hpp"

namespace ReSolve
{
  /**
   * @brief Constructor for PreconditionerUserMatrix.
   *
   * @param[in] matrix_handler - Pointer to the matrix handler
   */
  PreconditionerUserMatrix::PreconditionerUserMatrix(MatrixHandler* matrix_handler)
  {
    matrix_handler_ = matrix_handler;
    setMemorySpace();
  }

  /**
   * @brief Destructor for PreconditionerUserMatrix
   */
  PreconditionerUserMatrix::~PreconditionerUserMatrix()
  {
  }

  /**
   * @brief This preconditioner does not depend on A.
   */
  int PreconditionerUserMatrix::setup(matrix_type* /* A */)
  {
    return 0;
  }

  /**
   * @brief Set the explicit preconditioner matrix B.
   *
   * @param[in] B - Pointer to the preconditioning matrix
   */
  int PreconditionerUserMatrix::setPrecMatrix(matrix_type* B)
  {
    B_ = B;
    return 0;
  }

  /**
   * @brief Get the preconditioner matrix
   *
   * Necessary for some calculations
   *
   */
  matrix::Sparse* PreconditionerUserMatrix::getPrecMatrix()
  {
    return B_;
  }

  /**
   * @brief Applies the preconditioner depending on the side
   *
   * @param[in] rhs - Right-hand-side vector
   * @param[in] x   - Solution vector
   *
   * @return int 0 if successful, 1 if fails
   */
  int PreconditionerUserMatrix::apply(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    if (matrix_handler_ == nullptr || B_ == nullptr)
    {
      return 1;
    }

    matrix_handler_->matvec(B_, rhs, x, &ONE, &ZERO, memspace_);
    return 0;
  }

  void PreconditionerUserMatrix::setMemorySpace()
  {
    bool is_matrix_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_matrix_handler_hip  = matrix_handler_->getIsHipEnabled();

    if (is_matrix_handler_cuda || is_matrix_handler_hip)
    {
      memspace_ = memory::DEVICE;
    }
    else
    {
      memspace_ = memory::HOST;
    }
  }

} // namespace ReSolve
