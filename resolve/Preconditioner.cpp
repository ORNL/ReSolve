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

  Preconditioner::Side Preconditioner::getSide() const
  {
    return side_;
  }

  /**
   * @brief Set the preconditioning side
   *
   * @param[in] side - side of preconditioning
   * @return 0 if successful
   */
  int Preconditioner::setSide(Side side)
  {
    side_ = side;
    return 0;
  }

} // namespace ReSolve
