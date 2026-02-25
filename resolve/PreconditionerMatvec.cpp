/**
 * @file   PreconditionerMatvec.cpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief  Declaration of PreconditionerMatrix class.
 */

#include "PreconditionerMatvec.hpp"

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief Constructor for PreconditionerMatrix.
   *
   * @param[in] A - Pointer to the forward operator
   * @param[in] matrix_handler - Pointer to the matrix handler
   */
  PreconditionerMatvec::PreconditionerMatvec(matrix::Sparse* A, MatrixHandler* matrix_handler)
  {
    A_              = A;
    matrix_handler_ = matrix_handler;
    setMemorySpace();
  }

  /**
   * @brief Destructor for PreconditionerLU
   */
  PreconditionerMatvec::~PreconditionerMatvec()
  {
  }

  /**
   * @brief Set the preconditioning matrix
   *
   * @param[in] B - Pointer to the preconditioning matrix
   */
  int PreconditionerMatvec::setup(matrix::Sparse* B)
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
  matrix::Sparse* PreconditionerMatvec::getPrec()
  {
    return B_;
  }

  /**
   * @brief Setter for the side
   *
   * @param[in] The new side
   */
  int PreconditionerMatvec::setSide(std::string side)
  {
    if (side == "left" || side == "right")
    {
      side_ = side;
      return 0;
    }
    out::error() << "Choose either left (BA) or right (AB)\n";
    return 1;
  }

  /**
   * @brief getter for the side
   */
  std::string PreconditionerMatvec::getSide()
  {
    return side_;
  }

  /**
   * @brief Applies the preconditioner depending on the side
   *
   * @param[in] rhs - Right-hand-side vector
   * @param[in] x   - Solution vector
   *
   * @return int 0 if successful, 1 if fails
   */
  int PreconditionerMatvec::apply(vector_type* rhs, vector_type* x)
  {
    using namespace constants;
    matrix_handler_->matvec(B_, rhs, x, &ONE, &ZERO, memspace_);
    return 0;
  }

  void PreconditionerMatvec::setMemorySpace()
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
