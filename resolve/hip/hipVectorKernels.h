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
  void hip_set_array_const(index_type n, real_type c, real_type* v);
  void hipAddConst(real_type* array, real_type val, index_type n);
}
