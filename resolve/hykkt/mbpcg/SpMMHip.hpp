/**
 * @file SpMMHip.hpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief Sparse matrix-multivector multiplication implementation from https://github.com/hypre-space/hypre
 * 
 * This function is a modified copy of csr_spmv_device.c with all of its referenced functions and macros included. Some
 * non-essential code is removed. Since some small changes are required for HIP compatibility, the code is copied instead
 * of included as a dependency. If another solver needs this in the future, this can be integrated with VectorHandler and
 * potentially turned into SpGEMM.
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    class SpMMHip
    {
    public:
      int hypreDevice_CSRMatrixMatvec(matrix::Csr* A, vector::Vector* X, vector::Vector* result);
    };
  }
}