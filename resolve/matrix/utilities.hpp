#pragma once

namespace ReSolve
{
  namespace matrix
  {
    // Forward declarations
    class Coo;
    class Csr;

    /// @brief 
    /// @param A_coo 
    /// @param A_csr 
    /// @return 
    int coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr);
  }
}
