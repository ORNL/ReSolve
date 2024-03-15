/**
 * @file hipSketchingKernels.hip
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief HIP implementation of kernels for random sketching.
 * 
 * 
 */

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "hipSketchingKernels.h"

namespace ReSolve
{
  namespace hip
  {
    namespace kernels
    {
      /**
       * @brief For count sketch in sketching random method
       * 
       * @param[in]  n      -
       * @param[in]  k      -
       * @param[in]  labels -
       * @param[in]  flip   -
       * @param[in]  input  - 
       * @param[out] output -
       */
      __global__ void count_sketch_kernel(const index_type n,
                                          const index_type k, 
                                          const index_type* labels,
                                          const index_type* flip,
                                          const real_type* input,
                                          real_type* output)
      {
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        while (idx < n) {
          real_type val = input[idx];
          if (flip[idx] != 1){
            val *= -1.0;
          }    
          atomicAdd(&output[labels[idx]], val);
          idx += blockDim.x * gridDim.x;
        }
      }

      /**
       * @brief Walsh-Hadamard transform (select kernel)
       * 
       * @param[in]  k      -
       * @param[in]  perm   -
       * @param[in]  input  -
       * @param[out] output - 
       */
      __global__ void select_kernel(const index_type k, 
                                    const index_type* perm,
                                    const real_type* input,
                                    real_type* output)
      {
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x; 
        while (idx < k) {
          output[idx] = input[perm[idx]];
          idx += blockDim.x * gridDim.x;
        }
      }

      /**
       * @brief Walsh-Hadamard transform (scale)
       * 
       * @param[in]  n -
       * @param[in]  D -
       * @param[in]  x -
       * @param[out] y -
       */
      __global__ void scaleByD_kernel(const index_type n,
                                      const index_type* D,
                                      const real_type* x,
                                      real_type* y)
      {
        index_type idx = blockIdx.x * blockDim.x + threadIdx.x; 

        while (idx < n) {

          if (D[idx] == 1)
            y[idx] = x[idx];
          else
            y[idx] = (-1.0)*x[idx];

          idx += blockDim.x * gridDim.x;
        }
      }

      /**
       * @brief Single in-global memory radix-4 Fast Walsh Transform pass
       * (for strides exceeding elementary vector size).
       * 
       * @param d_Output - 
       * @param d_Input  -
       * @param stride   -
       */
      __global__ void fwtBatch2Kernel(real_type* d_Output, real_type* d_Input, index_type stride) 
      {
        const index_type pos = blockIdx.x * blockDim.x + threadIdx.x;
        const index_type N = blockDim.x * gridDim.x * 4;

        real_type* d_Src = d_Input + blockIdx.y * N;
        real_type* d_Dst = d_Output + blockIdx.y * N;

        index_type lo = pos & (stride - 1);
        index_type i0 = ((pos - lo) << 2) + lo;
        index_type i1 = i0 + stride;
        index_type i2 = i1 + stride;
        index_type i3 = i2 + stride;

        real_type D0 = d_Src[i0];
        real_type D1 = d_Src[i1];
        real_type D2 = d_Src[i2];
        real_type D3 = d_Src[i3];

        real_type T;
        T = D0;
        D0 = D0 + D2;
        D2 = T - D2;
        T = D1;
        D1 = D1 + D3;
        D3 = T - D3;
        T = D0;
        d_Dst[i0] = D0 + D1;
        d_Dst[i1] = T - D1;
        T = D2;
        d_Dst[i2] = D2 + D3;
        d_Dst[i3] = T - D3;
      }


      /**
       * @brief 
       * 
       * @param d_Output -
       * @param d_Input  -
       * @param log2N    -
       * 
       * @todo `d_Input` should be `const` parameter.
       * 
       */
      __global__ void fwtBatch1Kernel(real_type* d_Output, real_type* d_Input, index_type log2N) 
      {
        // Handle to thread block group

        cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
        const index_type N = 1 << log2N;
        const index_type base = blockIdx.x << log2N;

        //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
        extern __shared__ real_type s_data[];
        real_type* d_Src = d_Input + base;
        real_type* d_Dst = d_Output + base;

        for (index_type pos = threadIdx.x; pos < N; pos += blockDim.x) {
          s_data[pos] = d_Src[pos];
        }

        // Main radix-4 stages
        const index_type pos = threadIdx.x;

        for (index_type stride = N >> 2; stride > 0; stride >>= 2) {
          index_type lo = pos & (stride - 1);
          index_type i0 = ((pos - lo) << 2) + lo;
          index_type i1 = i0 + stride;
          index_type i2 = i1 + stride;
          index_type i3 = i2 + stride;

          cooperative_groups::sync(cta);
          real_type D0 = s_data[i0];
          real_type D1 = s_data[i1];
          real_type D2 = s_data[i2];
          real_type D3 = s_data[i3];

          real_type T;
          T = D0;
          D0 = D0 + D2;
          D2 = T - D2;
          T = D1;
          D1 = D1 + D3;
          D3 = T - D3;
          T = D0;
          s_data[i0] = D0 + D1;
          s_data[i1] = T - D1;
          T = D2;
          s_data[i2] = D2 + D3;
          s_data[i3] = T - D3;
        }

        // Do single radix-2 stage for odd power of two
        if (log2N & 1) {

          cooperative_groups::sync(cta);

          for (index_type pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
            index_type i0 = pos << 1;
            index_type i1 = i0 + 1;

            real_type D0 = s_data[i0];
            real_type D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
          }
        }

        cooperative_groups::sync(cta);

        for (index_type pos = threadIdx.x; pos < N; pos += blockDim.x) {
          d_Dst[pos] = s_data[pos];
        }
      }

    } // namespace kernels

    //
    // Kernel wrappers
    //

    /**
     * @brief Kernel wrapper for 
     * 
     * @param[in]  n      - 
     * @param[in]  k      - 
     * @param[in]  labels - 
     * @param[in]  flip   -
     * @param[in]  input  - 
     * @param[out] output - 
     * 
     * @todo Decide how to allow user to configure grid and block sizes.
     */
    void count_sketch_theta(index_type n,
                            index_type k,
                            index_type* labels,
                            index_type* flip,
                            real_type* input,
                            real_type* output)
    {
      hipLaunchKernelGGL(kernels::count_sketch_kernel,
                         dim3(10000),
                         dim3(1024),
                         0,
                         0,
                         n,
                         k,
                         labels,
                         flip,
                         input,
                         output);
    }

    /**
     * @brief Wrapper for `select` kernel, part of Walsh-Hadamard transform
     * 
     * @param[in]  k      -
     * @param[in]  perm   - 
     * @param[in]  input  -
     * @param[out] output - 
     * 
     * @todo Decide how to allow user to configure grid and block sizes.
     */
    void FWHT_select(index_type k,
                     index_type* perm,
                     real_type* input,
                     real_type* output)
    {
      hipLaunchKernelGGL(kernels::select_kernel, dim3(1000), dim3(1024), 0, 0,k, perm, input, output);
    }

    /**
     * @brief Wrapper for `scale` kernel, part of Walsh-Hadamard transform
     * 
     * @param[in]  n -
     * @param[in]  D -
     * @param[in]  x -
     * @param[out] y -
     * 
     * @todo Decide how to allow user to configure grid and block sizes.
     */
    void FWHT_scaleByD(index_type n,
                      index_type* D,
                      real_type* x,
                      real_type* y)
    {
      hipLaunchKernelGGL(kernels::scaleByD_kernel, dim3(1000), dim3(1024), 0, 0, n, D, x, y);
    }

    /**
     * @brief 
     * 
     * @param[in] M       -
     * @param[in] log2N   - 
     * @param[out] d_Data - 
     * 
     * @todo Decide if and how user should configure log2size, thread_n, etc.
     */
    void FWHT(index_type M, index_type log2N, real_type* d_Data)
    {
      const index_type ELEMENTARY_LOG2SIZE = 11;
      const index_type THREAD_N = 1024;
      index_type N = 1 << log2N;
      dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

      for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2) {
        hipLaunchKernelGGL(kernels::fwtBatch2Kernel, grid, dim3(THREAD_N), 0, 0,  d_Data, d_Data, N / 4);
      }

      hipLaunchKernelGGL(kernels::fwtBatch1Kernel, dim3(M), dim3(N / 4), N * sizeof(real_type), 0, d_Data, d_Data, log2N);
    }
  } // namespace hip
} // namespace ReSolve
