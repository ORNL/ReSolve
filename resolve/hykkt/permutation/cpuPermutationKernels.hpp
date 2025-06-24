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

      void mapIdx(int n, const int* perm, const double* old_val, double* new_val);
    }; // class CpuPermutationKernels
  } // namespace hykkt
} // namespace ReSolve
