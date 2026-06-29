#include "RandomizedConjugateGradientCuda.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    namespace kernels
    {
      template <index_type K>
      __global__ __launch_bounds__(256, 2) void SpMMTallSkinnyKernel(
          const index_type* __restrict__ A_row_ptr,
          const index_type* __restrict__ A_col_idx,
          const real_type* __restrict__ A_val,
          const real_type* __restrict__ B,
          real_type* __restrict__ result,
          index_type n)
      {
          const index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
          const index_type row = thread / 32;
          const index_type lane = thread % 32;

          if (2 * row >= n) return;

          index_type col_start = A_row_ptr[row];
          index_type col_end   = A_row_ptr[row + 1];

          real_type sum_top[K];
          real_type sum_bot[K];
          
          #pragma unroll
          for (int i = 0; i < K; ++i) {
              sum_top[i] = 0.0;
              sum_bot[i] = 0.0;
          }

          for (index_type j = col_start + lane; j < col_end; j += 32)
          {
              index_type col = A_col_idx[j];
              real_type a = A_val[j];

              #pragma unroll
              for (int i = 0; i < K; ++i)
              {
                  sum_top[i] += a * __ldg(&B[i * n + col]);
                  sum_bot[i] += a * __ldg(&B[i * n + (n - col - 1)]);
              }
          }

          #pragma unroll
          for (int i = 0; i < K; ++i)
          {
              for (index_type offset = 32 / 2; offset > 0; offset /= 2)
              {
                  sum_top[i] += __shfl_down_sync(0xffffffff, sum_top[i], offset);
                  sum_bot[i] += __shfl_down_sync(0xffffffff, sum_bot[i], offset);
              }
              
              if (lane == 0)
              {
                  result[i * n + row] = sum_top[i];
                  result[i * n + (n - row - 1)] = sum_bot[i];
              }
          }
      }
    } // namespace kernels
    
    RandomizedConjugateGradientCuda::RandomizedConjugateGradientCuda()
    {
      cusolverSpCreate(&cusolverHandle_);
      cusparseCreateMatDescr(&descrA_);
      cusolverSpCreateCsrcholInfo(&factorizationInfo_);
      buffer_ = nullptr;
    }

    RandomizedConjugateGradientCuda::~RandomizedConjugateGradientCuda()
    {
      cusolverSpDestroy(cusolverHandle_);
      cusparseDestroyMatDescr(descrA_);
      cusolverSpDestroyCsrcholInfo(factorizationInfo_);
      mem_.deleteOnDevice(buffer_);
    }

    int RandomizedConjugateGradientCuda::SpMMTallSkinny(matrix::Csr* A, const vector::Vector* X, vector::Vector* result)
    {
      index_type n = A->getNumRows();
      index_type k = X->getNumVectors();

      int       block_size = 256;
      int       num_blocks = (n * 32 / 2 + block_size - 1) / block_size;

      switch (k)
      {
      case 1:
        kernels::SpMMTallSkinnyKernel<1><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE),
                                                                A->getColData(memory::DEVICE),
                                                                A->getValues(memory::DEVICE),
                                                                X->getData(memory::DEVICE),
                                                                result->getData(memory::DEVICE),
                                                                n);
        break;
      case 2:
        kernels::SpMMTallSkinnyKernel<2><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE),
                                                                A->getColData(memory::DEVICE),
                                                                A->getValues(memory::DEVICE),
                                                                X->getData(memory::DEVICE),
                                                                result->getData(memory::DEVICE),
                                                                n);
        break;
      case 4:
        kernels::SpMMTallSkinnyKernel<4><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE),
                                                                A->getColData(memory::DEVICE),
                                                                A->getValues(memory::DEVICE),
                                                                X->getData(memory::DEVICE),
                                                                result->getData(memory::DEVICE),
                                                                n);
        break;
      case 8:
        kernels::SpMMTallSkinnyKernel<8><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE),
                                                                A->getColData(memory::DEVICE),
                                                                A->getValues(memory::DEVICE),
                                                                X->getData(memory::DEVICE),
                                                                result->getData(memory::DEVICE),
                                                                n);
        break;
      case 16:
        kernels::SpMMTallSkinnyKernel<16><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE),
                                                                A->getColData(memory::DEVICE),
                                                                A->getValues(memory::DEVICE),
                                                                X->getData(memory::DEVICE),
                                                                result->getData(memory::DEVICE),
                                                                n);
        break;
      default:
        return 1;
      }
      
      return 0;
    }
  } // namespace hykkt
} // namespace ReSolve
