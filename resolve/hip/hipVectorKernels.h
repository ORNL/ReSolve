/**
 * @file hipVectorKernels.h
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Contains declaration of HIP vector kernels.
 * @date 2023-12-08
 *
 * @note Kernel wrappers implemented here are intended for use in hardware
 * agnostic code.
 */
#pragma once

#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace hip
  {
    void setArrayConst(index_type n, real_type val, real_type* arr);
    void addConst(index_type n, real_type val, real_type* arr);
    void scale(index_type n, const real_type* diag, real_type* vec);
    void diagSolve(index_type n, const real_type* diag, real_type* vec);
    void max(index_type n, const real_type* x, const real_type* y, real_type* out);
    void abs(index_type n, const real_type* in, real_type* out);
  } // namespace hip
} // namespace ReSolve
