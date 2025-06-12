/**
 * @file cudaVectorKernels.cu
 * @author Slaven Peles (peless@ornl.gov), Shaked Regev (regevs@ornl.gov)
 * @brief Contains implementation of CUDA vector kernels and their wrappers.
 * @date 2023-12-08
 *
 * @note Kernel wrappers implemented here are intended for use in hardware
 * agnostic code.
 */
#include <cuda_runtime.h>

#include <resolve/cuda/cudaVectorKernels.h>
#include <resolve/cuda/cudaKernels.h>

namespace ReSolve
{
  namespace cuda {
    namespace kernels
    {

      /**
       * @brief CUDA kernel that sets values of an array to a constant.
       *
       * @param[in]  n   - length of the array
       * @param[in]  val - the value the array is set to
       * @param[out] arr - a pointer to the array
       *
       * @pre  `arr` is allocated to size `n`
       * @post `arr` elements are set to `val`
       */
      __global__ void set_const(index_type n, real_type val, real_type* arr)
      {
        index_type i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n)
        {
          arr[i] = val;
        }
      }

      /**
       * @brief CUDA kernel that adds a constant to each element of an array.
       *
       * @param[in]       n   - length of the array
       * @param[in]       val - the value to add to each element
       * @param[in, out]  arr - a pointer to the array
       *
       * @pre  `arr` is allocated to size `n`
       * @post `val` is added to each element of `arr`
       */
      __global__ void addConst(index_type n, real_type val, real_type* arr)
      {
        index_type i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n)
        {
          arr[i] += val;
        }
      }

      /**
       * @brief Scales a vector by a diagonal matrix represented by a vector
       *
       * @param[in]  n      - size of the vector
       * @param[in, out] vec - vector to be scaled. Changes in place.
       * @param[in]  d_val  - diagonal values
       *
       * @todo Decide how to allow user to configure grid and block sizes.
       */
      __global__ void scale(index_type n,
                                      const real_type* d_val,
                                      real_type* vec)
      {
        // Get the index of the element to be processed
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if the index is within bounds
        if (idx < n) {
          // Scale the vector element by the corresponding diagonal value
          vec[idx] *= d_val[idx];
        }
      }

    } // namespace kernels

    void setArrayConst(index_type n, real_type val, real_type* arr)
    {
      index_type num_blocks;
      index_type block_size = 512;
      num_blocks = (n + block_size - 1) / block_size;
      kernels::set_const<<<num_blocks, block_size>>>(n, val, arr);
    }

    void addConst(index_type n, real_type val, real_type* arr)
    {
      index_type num_blocks;
      index_type block_size = 512;
      num_blocks = (n + block_size - 1) / block_size;
      kernels::addConst<<<num_blocks, block_size>>>(n, val, arr);
    }

    /**
     * @brief Wrapper that scales a vector by a diagonal matrix
     *
     * @param[in]  n      - size of the vector
     * @param[in, out] vec - vector to be scaled. Changes in place.
     * @param[in]  d_val  - diagonal values
     *
     * @todo Decide how to allow user to configure grid and block sizes.
     */
    void scale(index_type n,
                        const real_type* diag,
                        real_type* vec)
    {
      // Define block size and number of blocks
      const int block_size = 256;
      int num_blocks = (n + block_size - 1) / block_size;
      // Launch the kernel
      kernels::scale<<<num_blocks, block_size>>>(n, diag, vec);
    }
  } // namespace cuda
} // namespace ReSolve
