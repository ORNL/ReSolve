/**
 * @file hipKernels.hip
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief 
 * @date 2023-12-08
 * 
 * 
 */

#include "hipKernels.h"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace ReSolve {
  namespace kernels {

    /**
     * @brief Computes v^T * [u1 u2] where v is n x k multivector
     * and u1 and u2 are n x 1 vectors.
     * 
     * @tparam Tv5 - Size of shared memory
     *  
     * @param[in] u1      - (n x 1) vector
     * @param[in] u2      - (n x 1) vector
     * @param[in] v       - (n x k) multivector
     * @param[out] result - (k x 2) multivector
     * @param[in] k       - dimension of the subspace
     * @param[in] N       - size of vectors u1, u2
     */
    template <size_t Tv5 = 1024>
    __global__ void MassIPTwoVec_kernel(const real_type* __restrict__ u1, 
                                        const real_type* __restrict__ u2, 
                                        const real_type* __restrict__ v, 
                                        real_type* result,
                                        const index_type k, 
                                        const index_type N)
    {
      index_type t = threadIdx.x;
      index_type bsize = blockDim.x;

      // assume T threads per thread block (and k reductions to be performed)
      volatile __shared__ real_type s_tmp1[Tv5];

      volatile __shared__ real_type s_tmp2[Tv5];
      // map between thread index space and the problem index space
      index_type j = blockIdx.x;
      s_tmp1[t] = 0.0;
      s_tmp2[t] = 0.0;
      index_type nn = t;
      real_type can1, can2, cbn;

      while (nn < N) {
        can1 = u1[nn];
        can2 = u2[nn];

        cbn = v[N * j + nn];
        s_tmp1[t] += can1 * cbn;
        s_tmp2[t] += can2 * cbn;

        nn += bsize;
      }

      __syncthreads();

      if (Tv5 >= 1024) {
        if(t < 512) {
          s_tmp1[t] += s_tmp1[t + 512];
          s_tmp2[t] += s_tmp2[t + 512];
        }
        __syncthreads();
      }
      if (Tv5 >= 512) {
        if(t < 256) {
          s_tmp1[t] += s_tmp1[t + 256];
          s_tmp2[t] += s_tmp2[t + 256];
        }
        __syncthreads();
      }
      {
        if (t < 128) {
          s_tmp1[t] += s_tmp1[t + 128];
          s_tmp2[t] += s_tmp2[t + 128];
        }
        __syncthreads();
      }
      {
        if (t < 64) {
          s_tmp1[t] += s_tmp1[t + 64];
          s_tmp2[t] += s_tmp2[t + 64];
        }
        __syncthreads();
      }

      if (t < 32) {
        s_tmp1[t] += s_tmp1[t + 32];
        s_tmp2[t] += s_tmp2[t + 32];

        s_tmp1[t] += s_tmp1[t + 16];
        s_tmp2[t] += s_tmp2[t + 16];

        s_tmp1[t] += s_tmp1[t + 8];
        s_tmp2[t] += s_tmp2[t + 8];

        s_tmp1[t] += s_tmp1[t + 4];
        s_tmp2[t] += s_tmp2[t + 4];

        s_tmp1[t] += s_tmp1[t + 2];
        s_tmp2[t] += s_tmp2[t + 2];

        s_tmp1[t] += s_tmp1[t + 1];
        s_tmp2[t] += s_tmp2[t + 1];
      }
      if (t == 0) {
        result[blockIdx.x] = s_tmp1[0];
        result[blockIdx.x + k] = s_tmp2[0];
      }
    }


    /**
     * @brief AXPY y = y - x*alpha where alpha is [k x 1], needed in 1 and 2 synch GMRES
     * 
     * @tparam Tmaxk 
     * 
     * @param[in]  N      -
     * @param[in]  k      -
     * @param[in]  x_data -
     * @param[out] y_data -
     * @param[in]  alpha  -
     */
    template <size_t Tmaxk = 1024>
    __global__ void massAxpy3_kernel(index_type N,
                                    index_type k,
                                    const real_type* x_data,
                                    real_type* y_data,
                                    const real_type* alpha)
    {
      index_type i = blockIdx.x * blockDim.x + threadIdx.x;
      index_type t = threadIdx.x;

      __shared__ real_type s_alpha[Tmaxk];
      if (t < k) {
        s_alpha[t] = alpha[t];
      }
      __syncthreads();
      while (i < N) {
        real_type temp = 0.0;
        for(index_type j = 0; j < k; ++j) {
          temp += x_data[j * N + i] * s_alpha[j];
        }
        y_data[i] -= temp;
        i += (blockDim.x*gridDim.x);
      }
    }

    /**
     * @brief Pass through matrix rows and sum each as \sum_{j=1}^m abs(a_{ij})
     * 
     * @param[in]  n      -
     * @param[in]  nnz    -
     * @param[in]  a_ia   -
     * @param[in]  a_val  -
     * @param[out] result -
     */
    __global__ void matrixInfNormPart1(const index_type n, 
                                       const index_type nnz, 
                                       const index_type* a_ia,
                                       const real_type* a_val, 
                                       real_type* result)
    {
      index_type idx = blockIdx.x*blockDim.x + threadIdx.x;
      while (idx < n) {
        real_type sum = 0.0;
        for (index_type i = a_ia[idx]; i < a_ia[idx+1]; ++i) {
          sum = sum + fabs(a_val[i]);
        }
        result[idx] = sum;
        idx += (blockDim.x * gridDim.x);
      }
    }

    /**
     * @brief 
     * 
     * @param[in]  n      - vector size
     * @param[in]  input  -
     * @param[out] result - 
     */
    __global__ void vectorInfNorm(const index_type n, 
                                  const real_type* input, 
                                  real_type* result)
    {

      index_type idx = blockIdx.x * blockDim.x + threadIdx.x;

      volatile __shared__ real_type s_max[1024];
      index_type t = threadIdx.x;
      index_type bsize = blockDim.x;
      real_type local_max = 0.0;
      if (idx < n) {
        local_max = fabs(input[idx]);
      }

      idx += (blockDim.x*gridDim.x);
      
      while (idx < n) {
        local_max = fmax(fabs(input[idx]), local_max);
        idx += (blockDim.x * gridDim.x);
      }
      s_max[t] = local_max;
      __syncthreads();

      // reduction

      if (bsize >= 1024) {
        if(t < 512) {
          s_max[t] = fmax(s_max[t], s_max[t + 512]);
        }
        __syncthreads();
      }
      if (bsize >= 512) {
        if(t < 256) {
          s_max[t] = fmax(s_max[t], s_max[t + 256]);
        }
        __syncthreads();
      }

      if (bsize >= 256) {
        if(t < 128) {
          s_max[t] = fmax(s_max[t], s_max[t + 128]);
        }
        __syncthreads();
      }

      if (bsize >= 128) {
        if(t < 64) {
          s_max[t] = fmax(s_max[t], s_max[t + 64]);
        }
        __syncthreads();
      }
      //unroll for last warp
      if (t < 64) {
        s_max[t] = fmax(s_max[t], s_max[t + 64]);
        s_max[t] = fmax(s_max[t], s_max[t + 32]);
        s_max[t] = fmax(s_max[t], s_max[t + 16]);
        s_max[t] = fmax(s_max[t], s_max[t + 8]);
        s_max[t] = fmax(s_max[t], s_max[t + 4]);
        s_max[t] = fmax(s_max[t], s_max[t + 2]);
        s_max[t] = fmax(s_max[t], s_max[t + 1]);

      }
      if (t == 0) {
        index_type bid = blockIdx.x;
        index_type gid = gridDim.x;
        result[blockIdx.x] = s_max[0];
      }
    }


    /**
     * @brief 
     * 
     * @param n           - vector size
     * @param perm_vector - permutation map
     * @param vec_in      - input vector
     * @param vec_out     - permuted vector
     */
    __global__ void permuteVectorP_kernel(const index_type n, 
                                          const index_type* perm_vector,
                                          const real_type* vec_in, 
                                          real_type* vec_out)
    {
      //one thread per vector entry, pass through rows
      index_type idx = blockIdx.x*blockDim.x + threadIdx.x;
      while (idx < n) {
        vec_out[idx] = vec_in[perm_vector[idx]];
        idx += (blockDim.x * gridDim.x);
      }
    }

    /**
     * @brief 
     * 
     * @param n           - vector size
     * @param perm_vector - permutation map
     * @param vec_in      - input vector
     * @param vec_out     - permuted vector
     */
    __global__ void permuteVectorQ_kernel(const index_type n, 
                                          const index_type* perm_vector,
                                          const real_type* vec_in, 
                                          real_type* vec_out)
    {
      //one thread per vector entry, pass through rows
      index_type idx = blockIdx.x*blockDim.x + threadIdx.x;
      while (idx < n) {
        vec_out[perm_vector[idx]] = vec_in[idx];
        idx += (blockDim.x * gridDim.x);
      }
    }

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
    __global__ void  count_sketch_kernel(const index_type n,
                                         const index_type k, 
                                         const index_type* labels,
                                         const index_type* flip,
                                         const real_type* input,
                                         real_type* output){

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
                                  real_type* output){

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
                                    real_type* y){
      index_type idx = blockIdx.x * blockDim.x + threadIdx.x; 

      while (idx < n){

        if (D[idx] == 1) y[idx]=x[idx];
        else y[idx]= (-1.0)*x[idx];

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
   * @brief Computes result = mvec^T * [vec1 vec2]
   * 
   * @param n      - size of vectors vec1, vec2
   * @param i      - 
   * @param vec1   - (n x 1) vector 
   * @param vec2   - (n x 1) vector
   * @param mvec   - (n x (i+1)) multivector
   * @param result - ((i+1) x 2) multivector
   * 
   * @todo Input data should be `const`.
   * @todo Is it coincidence that the block size is equal to the default
   * value of Tv5?
   * @todo Should we use dynamic shared memory here instead?
   */
  void mass_inner_product_two_vectors(index_type n, 
                                      index_type i, 
                                      real_type* vec1, 
                                      real_type* vec2, 
                                      real_type* mvec, 
                                      real_type* result)
  {
    hipLaunchKernelGGL(kernels::MassIPTwoVec_kernel,
                       dim3(i + 1),
                       dim3(1024),
                       0,
                       0,
                       vec1,
                       vec2,
                       mvec,
                       result,
                       i + 1,
                       n);
  }

  /**
   * @brief Computes y := y - x*alpha
   * 
   * @param[in]  n     - vector size
   * @param[in]  i     - number of vectors in the multivector
   * @param[in]  x     - (n x (i+1)) multivector 
   * @param[out] y     - (n x (i+1)) multivector
   * @param[in]  alpha - ((i+1) x 1) vector
   */
  void mass_axpy(index_type n, index_type i, real_type* x, real_type* y, real_type* alpha)
  {
    hipLaunchKernelGGL(kernels::massAxpy3_kernel,
                       dim3((n + 384 - 1) / 384),
                       dim3(384),
                       0,
                       0,
                       n,
                       i,
                       x,
                       y,
                       alpha);
  }

  /**
   * @brief 
   * 
   * @param[in]  n      -
   * @param[in]  nnz    -
   * @param[in]  a_ia   - 
   * @param[in]  a_val  -
   * @param[out] result -
   * 
   * @todo Decide how to allow user to configure grid and block sizes.
   */
  void matrix_row_sums(index_type n, 
                       index_type nnz, 
                       index_type* a_ia,
                       real_type* a_val, 
                       real_type* result)
  {
    hipLaunchKernelGGL(kernels::matrixInfNormPart1, dim3(1000), dim3(1024), 0, 0, n, nnz, a_ia, a_val, result);
  }

  /**
   * @brief 
   * 
   * @param n      -
   * @param input  -
   * @param buffer - 
   * @param result - 
   * 
   * @todo Decide how to allow user to configure grid and block sizes.
   */
  void vector_inf_norm(index_type n,  
                       real_type* input,
                       real_type* buffer, 
                       real_type* result)
  {
    hipLaunchKernelGGL(kernels::vectorInfNorm, dim3(1024), dim3(1024), 0, 0, n, input, buffer);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(kernels::vectorInfNorm, dim3(1), dim3(1024), 0, 0, 1024, buffer, buffer);
    hipMemcpy(result, buffer, sizeof(real_type) * 1, hipMemcpyDeviceToHost);
  }

  /**
   * @brief 
   * 
   * @param n           - vector size
   * @param perm_vector - permutation map
   * @param vec_in      - input vector
   * @param vec_out     - permuted vector
   */
  void permuteVectorP(index_type n, 
                      index_type* perm_vector,
                      real_type* vec_in, 
                      real_type* vec_out)
  {
    hipLaunchKernelGGL(kernels::permuteVectorP_kernel,
                       dim3(1000),
                       dim3(1024),
                       0,
                       0,
                       n,
                       perm_vector,
                       vec_in,
                       vec_out);
  }

  /**
   * @brief 
   * 
   * @param n           - vector size
   * @param perm_vector - permutation map
   * @param vec_in      - input vector
   * @param vec_out     - permuted vector
   */
  void permuteVectorQ(index_type n, 
                      index_type* perm_vector,
                      real_type* vec_in, 
                      real_type* vec_out)
  {
    hipLaunchKernelGGL(kernels::permuteVectorQ_kernel,
                       dim3(1000),
                       dim3(1024),
                       0,
                       0,
                       n,
                       perm_vector,
                       vec_in,
                       vec_out);
  }

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
  void  count_sketch_theta(index_type n,
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

} // namespace ReSolve
