#include <resolve/Common.hpp>
#include <resolve/vector/VectorKernels.hpp>
#include <hip/hip_runtime.h>

namespace ReSolve { namespace vector {

namespace kernels {

  __global__ void set_const(index_type n, real_type val, real_type* arr)
  {
    index_type i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n)
    {
      arr[i] = val;
      i += blockDim.x * gridDim.x;
    }
  }
} // namespace kernels

void set_array_const(index_type  n, real_type val, real_type* arr)
{
   index_type num_blocks;
   index_type block_size = 512;
   num_blocks = (n + block_size - 1) / block_size;
   hipLaunchKernelGGL( kernels::set_const, dim3(num_blocks), dim3(block_size), 0, 0, n, val, arr);
}

}} // namespace ReSolve::vector
