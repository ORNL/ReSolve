/**
 * @file hipSketchingKernels.h
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Contains prototypes of HIP random sketching kernels.
 * @date 2023-12-08
 * 
 * @note These kernels will be used in HIP specific code, only.
 * 
 */

#pragma once

#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace hip
  {
    // needed for rand solver
    void  count_sketch_theta(index_type n,
                            index_type k,
                            index_type* labels,
                            index_type* flip,
                            real_type* input,
                            real_type* output);

    void FWHT_select(index_type k,
                    index_type* perm,
                    real_type* input,
                    real_type* output);

    void FWHT_scaleByD(index_type n,
                      index_type* D,
                      real_type* x,
                      real_type* y);

    void FWHT(index_type M, index_type log2N, real_type* d_Data); 
  } // namespace hip
} // namespace ReSolve
