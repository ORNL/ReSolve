/**
 * @file HipPermutationKernels.hpp
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
     * @brief HyKKT permutation kernels for HIP.
     */
    class HipPermutationKernels : public PermutationKernelsImpl
    {
    public:
      HipPermutationKernels()
      {
      }

      virtual ~HipPermutationKernels()
      {
      }

      void mapIdx(index_type n, const index_type* perm, const real_type* old_val, real_type* new_val);
    }; // class HipPermutationKernels
  } // namespace hykkt
} // namespace ReSolve
