#pragma once

#include <list>
#include <resolve/MemoryUtils.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve
{
  namespace matrix
  {
    // Forward declarations
    class Coo;
    class Csr;

    /// @brief Converts row-major sorted COO to CSR matrix, preserves symmetry properties.
    int coo2csr_simple(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace);
  }
}
