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

} // namespace ReSolve
