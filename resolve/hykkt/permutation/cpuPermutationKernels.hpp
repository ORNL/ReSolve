/**
 * @file cpuPermutationKernels.hpp
 * @author Shaked Regev (regevs@ornl.gov)
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
     * @brief Declaration of PermutationKernelsImpl for CPU.
     */
    class CpuPermutationKernels : public PermutationKernelsImpl
    {
    public:
      CpuPermutationKernels()
      {
      }

      virtual ~CpuPermutationKernels()
      {
      }

      void mapIdx(index_type n, const index_type* perm, const real_type* old_val, real_type* new_val);
    }; // class CpuPermutationKernels
  } // namespace hykkt
} // namespace ReSolve
