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

  //needed for matrix inf nrm
  void matrix_row_sums(index_type n, 
                       index_type nnz, 
                       const index_type* a_ia,
                       const real_type* a_val, 
                       real_type* result);

  // needed for rand solver
  void  count_sketch_theta(index_type n,
                           index_type k,
                           const index_type* labels,
                           const index_type* flip,
                           const real_type* input,
                           real_type* output);

  void FWHT_select(index_type k,
                   const index_type* perm,
                   const real_type* input,
                   real_type* output);

  void FWHT_scaleByD(index_type n,
                     const index_type* D,
                     const real_type* x,
                     real_type* y);

  void FWHT(index_type M, index_type log2N, real_type* d_Data); 

} // namespace ReSolve
