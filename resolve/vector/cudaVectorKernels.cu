#include "cudaVectorKernels.h"
#include "VectorKernels.hpp"


namespace ReSolve { namespace vector {

namespace kernels {

__global__ void set_const(index_type n, real_type val, real_type* arr)
{
  index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
  {
    arr[i] = val;
  }
}

} // namespace kernels

void set_array_const(index_type n, real_type val, real_type* arr)
{
  index_type num_blocks;
  index_type block_size = 512;
  num_blocks = (n + block_size - 1) / block_size;
  kernels::set_const<<<num_blocks, block_size>>>(n, val, arr);
}

}} // namespace ReSolve::vector