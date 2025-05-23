/**
 * @file hipKernels.h
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Contains prototypes of HIP kernels.
 * @date 2023-12-08
 *
 * @note These kernels will be used in HIP specific code, only.
 *
 */

#pragma once

#include <resolve/Common.hpp>

namespace ReSolve {

  void mass_inner_product_two_vectors(index_type n,
                                      index_type i,
                                      real_type* vec1,
                                      real_type* vec2,
                                      real_type* mvec,
                                      real_type* result);
  void mass_axpy(index_type n, index_type i, real_type* x, real_type* y, real_type* alpha);

  void LeftDiagScale(index_type n,
                     const index_type* a_row_ptr,
                     real_type* a_val,
                     const real_type* diag);

  void hipRightDiagScale(index_type n,
                      const index_type* a_row_ptr,
                      const index_type* a_col_idx,
                      real_type* a_val,
                      const real_type* diag);

  //needed for matrix inf nrm
  void matrix_row_sums(index_type n,
                       index_type nnz,
                       index_type* a_ia,
                       real_type* a_val,
                       real_type* result);

  // needed for triangular solve

  void permuteVectorP(index_type n,
                      index_type* perm_vector,
                      real_type* vec_in,
                      real_type* vec_out);

  void permuteVectorQ(index_type n,
                      index_type* perm_vector,
                      real_type* vec_in,
                      real_type* vec_out);


  void vector_inf_norm(index_type n,
                       real_type* input,
                       real_type * buffer,
                       real_type* result);
} // namespace ReSolve
