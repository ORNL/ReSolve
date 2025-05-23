/**
 * @file cudaKernels.h
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 *
 * @brief Contains prototypes of CUDA kernels.
 *
 * @note These kernels will be used in CUDA specific code, only.
 *
 */
#pragma once

#include <resolve/Common.hpp>

namespace ReSolve {

  void mass_inner_product_two_vectors(index_type n,
                                      index_type i,
                                      const real_type* vec1,
                                      const real_type* vec2,
                                      const real_type* mvec,
                                      real_type* result);

  void mass_axpy(index_type n, index_type i, const real_type* x, real_type* y, const real_type* alpha);

  void LeftDiagScale(index_type n,
                      const index_type* a_row_ptr,
                      real_type* a_val,
                      const real_type* diag);

  void cudaRightDiagScale(index_type n,
                       const index_type* a_row_ptr,
                       const index_type* a_col_idx,
                       real_type* a_val,
                       const real_type* diag);

  //needed for matrix inf nrm
  void matrix_row_sums(index_type n,
                       index_type nnz,
                       const index_type* a_ia,
                       const real_type* a_val,
                       real_type* result);

} // namespace ReSolve
