#pragma once

#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  namespace matrix
  {
    // Forward declarations
    class Coo;
    class Csr;

    /// @brief Converts symmetric or general COO to general CSR matrix
    int coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace);

    /// @brief Converts symmetric or general COO to general CSR matrix
    int coo2csr_new(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace);
  }
}
