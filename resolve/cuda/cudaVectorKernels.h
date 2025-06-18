/**
 * @file cudaVectorKernels.h
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Contains declarations of CUDA vector kernel wrappers.
 * @date 2023-12-08
 *
 * @note Kernel wrappers implemented here are intended for use in hardware
 * agnostic code.
 */
#pragma once

#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace cuda
  {
    void setArrayConst(index_type n, real_type val, real_type* arr);
    void addConst(index_type n, real_type val, real_type* arr);
    void scale(index_type n, const real_type* diag, real_type* vec);
  } // namespace cuda
} // namespace ReSolve
