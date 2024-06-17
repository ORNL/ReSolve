#pragma once

#include "Coo.hpp"
#include "Csr.hpp"

namespace ReSolve
{
  namespace matrix
  {
    int expand(Coo&);
    int expand(Csr&);
  } // namespace matrix
} // namespace ReSolve
