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

    } // namespace kernels

    void cuda_set_array_const(index_type n, real_type val, real_type* arr)
    {
      index_type num_blocks;
      index_type block_size = 512;
      num_blocks = (n + block_size - 1) / block_size;
      kernels::set_const<<<num_blocks, block_size>>>(n, val, arr);
    }

    void cudaAddConst(index_type n, real_type val, real_type* arr)
    {
      index_type num_blocks;
      index_type block_size = 512;
      num_blocks = (n + block_size - 1) / block_size;
      kernels::addConst<<<num_blocks, block_size>>>(n, val, arr);
    }
  } // namespace cuda
} // namespace ReSolve
