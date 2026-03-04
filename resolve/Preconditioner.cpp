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

  std::string Preconditioner::getSide() const
  {
    return side_;
  }

  /**
   * @brief Set the preconditioning side
   *
   * @param[in] side - "left" or "right"
   * @return 0 if successful, 1 if invalid value
   */
  int Preconditioner::setSide(const std::string& side)
  {
    if (side == "left" || side == "right") {
      side_ = side;
      return 0;
    }
    return 1;
  }

} // namespace ReSolve
