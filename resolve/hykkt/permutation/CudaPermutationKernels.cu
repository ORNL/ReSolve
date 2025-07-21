/**
 * @file CudaPermutationKernels.cu
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 *
 */

#include <cuda_runtime.h>

#include "CudaPermutationKernels.hpp"

namespace ReSolve
{
  namespace hykkt
  {
    /**
     * @brief Kernel to map indices for CUDA.
     * This kernel is used to permute the values in old_val based on the perm array.
     *
     * @param[in] n       - matrix size (number of elements to permute)
     * @param[in] perm    - desired permutation array (indices for new_val)
     * @param[in] old_val - the array containing the original values
     * @param[out] new_val - the array to store the permuted values
     */
    __global__ void mapIdxKernel(index_type n, const index_type* perm, const real_type* old_val, real_type* new_val)
    {
      index_type i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
      {
        new_val[i] = old_val[perm[i]];
      }
    }

    /**
     * @brief Maps the values in old_val to new_val based on perm for CUDA.
     * This is the concrete implementation of the pure virtual method.
     *
     * @param[in] n       - matrix size (number of elements to permute)
     * @param[in] perm    - desired permutation array (indices for new_val)
     * @param[in] old_val - the array containing the original values
     * @param[out] new_val - the array to store the permuted values
     *
     * @pre n is a positive integer, perm is an array of 0 to n-1
     * (in some order), old_val is initialized.
     * @post new_val contains the permuted old_val, where new_val[i] = old_val[perm[i]].
     */
    void CudaPermutationKernels::mapIdx(index_type n, const index_type* perm, const real_type* old_val, real_type* new_val)
    {
      // Launch the CUDA kernel
      index_type blockSize = 256;
      index_type numBlocks = (n + blockSize - 1) / blockSize;
      mapIdxKernel<<<numBlocks, blockSize>>>(n, perm, old_val, new_val);
    }
  } // namespace hykkt
} // namespace ReSolve
