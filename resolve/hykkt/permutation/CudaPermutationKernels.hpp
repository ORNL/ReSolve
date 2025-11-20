/**
 * @file CudaPermutationKernels.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 *
 */

#pragma once

#include "PermutationKernelsImpl.hpp"

namespace ReSolve
{
  namespace hykkt
  {
    /**
     * @brief HyKKT permutation kernels for CUDA.
     */
    class CudaPermutationKernels : public PermutationKernelsImpl
    {
    public:
      CudaPermutationKernels()
      {
      }

      virtual ~CudaPermutationKernels()
      {
      }

      void mapIdx(index_type n, const index_type* perm, const real_type* old_val, real_type* new_val);
    }; // class CudaPermutationKernels
  } // namespace hykkt
} // namespace ReSolve
