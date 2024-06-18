#pragma once

#include "Coo.hpp"
#include "Csr.hpp"

namespace ReSolve
{
  namespace matrix
  {
    /// @brief Expands a symmetric COO matrix stored only as a half to its full size
    ///
    /// If the matrix has not been deduplicated, the result is undefined
    int expand(Coo&);

    /// @brief Expands a symmetric CSR matrix stored only as a half to its full size
    ///
    /// If the matrix has not been deduplicated, the result is undefined
    int expand(Csr&);
  } // namespace matrix
} // namespace ReSolve
