#pragma once

#include "Coo.hpp"
#include "Csc.hpp"
#include "Csr.hpp"

namespace ReSolve
{
  namespace matrix
  {
    /// @brief Expands a symmetric COO matrix stored only as a half to its full size
    int expand(Coo&);

    /// @brief Expands a symmetric CSR matrix stored only as a half to its full size
    int expand(Csr&);

    /// @brief Expands a symmetric CSR matrix stored only as a half to its full size
    int expand(Csc&);
  } // namespace matrix
} // namespace ReSolve
