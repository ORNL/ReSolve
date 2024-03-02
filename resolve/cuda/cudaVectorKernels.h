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
  void cuda_set_array_const(index_type n, real_type val, real_type* arr);
}