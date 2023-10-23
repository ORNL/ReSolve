#pragma once

#include <cuda_runtime.h>
// #include <cusparse.h>
// #include <cublas_v2.h>
// #include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include <resolve/Common.hpp>

//***************************************************************************//
//**** See VectorKernels.hpp for kernel wrapper functions documentation  ****//
//***************************************************************************//

namespace ReSolve { namespace vector {

namespace kernels {
  // __global__ void adapt_diag_scale(index_type, index_type, real_type*, index_type*, index_type*, real_type*, index_type*,
  //    index_type*, real_type*, index_type*, index_type*, real_type*, real_type*, real_type*, real_type*);

  // __global__ void adapt_row_max(index_type, index_type, real_type*, index_type*, index_type*, real_type*, index_type*,
  //    index_type*, real_type*, index_type*, index_type*, real_type*);

  // __global__ void add_const(index_type, index_type, index_type*);

  /**
   * @brief CUDA kernel that sets values of an array to a constant.
   *
   * @param[in]  n   - length of the array
   * @param[in]  val - the value the array is set to
   * @param[out] arr - a pointer to the array
   * 
   * @pre  `arr` is allocated to size `n`
   * @post `arr` elements are set to `val`
   */
  __global__ void set_const(index_type n, real_type val, real_type* arr);

  // __global__ void add_vecs(index_type, real_type*, real_type, real_type*);

  // __global__ void mult_const(index_type, real_type, real_type*);

  // __global__ void add_diag(index_type, real_type, index_type*, index_type*, real_type*);

  // __global__ void inv_vec_scale(index_type, real_type*, real_type*);

  // __global__ void vec_scale(index_type, real_type*, real_type*);

  // __global__ void concatenate(index_type, index_type, index_type, index_type, real_type*, index_type*, index_type*,
  //   real_type*, index_type*, index_type*, real_type*, index_type*, index_type*);

  // __global__ void row_scale(index_type, real_type*, index_type*, index_type*, real_type*, real_type*,
  //     real_type*, real_type*);

  // __global__ void diag_scale(index_type, index_type, real_type*, index_type*, index_type*, real_type*, index_type*,
  //   index_type*, real_type*, real_type*, real_type*, index_type);

  // __global__ void row_max(index_type, index_type, real_type*, index_type*, index_type*, real_type*, index_type*, index_type*,
  //    real_type* scale);
} // namespace kernels

}} // namespace ReSolve::vector