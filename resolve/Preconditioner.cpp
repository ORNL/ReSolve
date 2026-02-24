/**
 * @file   Preconditioner.cpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Implementation of preconditioner base class.
 *
 */

#include "Preconditioner.hpp"

namespace ReSolve
{
  Preconditioner::Preconditioner()
  {
  }

  Preconditioner::~Preconditioner()
  {
  }

  int Preconditioner::reset(matrix_type* /* A */)
  {
    return 1;
  }

  std::string Preconditioner::getSide()
  {
    return side_;
  }

  /**
   * @brief Used to get the preconditioning matrix for Preconditioner Matvec
   *
   * Should not be called unless using PreconditionerMatvec
   */
  matrix::Sparse* Preconditioner::getPrec()
  {
    return nullptr;
  }

} // namespace ReSolve
