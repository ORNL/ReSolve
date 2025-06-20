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
     * @brief Declaration of PermutationKernelsImpl for CUDA.
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

      void mapIdx(int n, const int* perm, const double* old_val, double* new_val);
    }; // class CudaPermutationKernels
  } // namespace hykkt
} // namespace ReSolve
