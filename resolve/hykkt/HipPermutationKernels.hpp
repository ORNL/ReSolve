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
     * @brief Declaration of PermutationKernelsImpl for HIP.
     */
    class HipPermutationKernels : public PermutationKernelsImpl
    {
    public:
      HipPermutationKernels()
      {}
      virtual ~HipPermutationKernels()
      {}

      void mapIdx(int n, const int* perm, const double* old_val, double* new_val);
    }; // class HipPermutationKernels
  } // namespace hykkt
} // namespace ReSolve