/**
 * @file cudaSketchingKernels.h
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 *
 * @brief Contains prototypes of CUDA random sketching kernels.
 *
 * @note These kernels will be used in CUDA specific code, only.
 *
 */
#pragma once

#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace cuda
  {
    // needed for rand solver
    void count_sketch_theta(index_type        n,
                            index_type        k,
                            const index_type* labels,
                            const index_type* flip,
                            const real_type*  input,
                            real_type*        output);

    void FWHT_select(index_type        k,
                     const index_type* perm,
                     const real_type*  input,
                     real_type*        output);

    void FWHT_scaleByD(index_type        n,
                       const index_type* D,
                       const real_type*  x,
                       real_type*        y);

    void FWHT(index_type M, index_type log2N, real_type* d_Data);
  } // namespace cuda
} // namespace ReSolve
