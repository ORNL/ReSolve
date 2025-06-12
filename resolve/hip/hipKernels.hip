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
  namespace hip {
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
      * @brief Scales a csr matrix on the left by a diagonal matrix
      *
      * @param[in]  n      - number of rows in the matrix
      * @param[in]  a_row_ptr - row pointers (CSR storage)
      * @param[in, out]  a_val    - values (CSR storage). Changes in place.
      * @param[in]  d_val    - diagonal values
      *
      * @todo Decide how to allow user to configure grid and block sizes.
      */
      __global__ void leftScale(index_type n,
                                    const index_type* a_row_ptr,
                                    real_type* a_val,
                                    const real_type* d_val)
      {
        // Get row index from thread and block indices
        index_type row = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if the thread's row is within matrix bounds
        if (row < n) {
          // Get the start and end positions for this row in the CSR format
          index_type row_start = a_row_ptr[row];
          index_type row_end = a_row_ptr[row + 1];

          // Get the scaling factor for this row from the diagonal matrix
          real_type scale = d_val[row];

          // Scale all non-zero elements in this row
          for (index_type i = row_start; i < row_end; i++) {
            a_val[i] *= scale;
          }
        }
      }

      /**
      * @brief Scales a csr matrix on the right by a diagonal matrix
      *
      * @param[in]  n      - number of rows in the matrix
      * @param[in]  a_row_ptr - row pointers (CSR storage)
      * @param[in]  a_col_ind - column indices (CSR storage)
      * @param[in, out]  a_val    - values (CSR storage). Changes in place.
      * @param[in]  d_val    - diagonal values
      *
      * @todo Decide how to allow user to configure grid and block sizes.
      */
      __global__ void rightScale(index_type n,
                                    const index_type* a_row_ptr,
                                    const index_type* a_col_ind,
                                    real_type* a_val,
                                    const real_type* d_val)
      {
        // Get row index from thread and block indices
        index_type row = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if the thread's row is within matrix bounds
        if (row < n) {
          // Get the start and end positions for this row in the CSR format
          index_type row_start = a_row_ptr[row];
          index_type row_end = a_row_ptr[row + 1];

          // Scale all non-zero elements in this row
          for (index_type i = row_start; i < row_end; i++) {
            a_val[i] *= d_val[a_col_ind[i]];
          }
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
                        dim3(i),
                        dim3(1024),
                        0,
                        0,
                        vec1,
                        vec2,
                        mvec,
                        result,
                        i,
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
    * @brief Wrapper that scales a csr matrix on the left by a diagonal matrix
    *
    * @param[in]  n      - number of rows in the matrix
    * @param[in]  a_row_ptr - row pointers (CSR storage)
    * @param[in, out]  a_val    - values (CSR storage). Changes in place.
    * @param[in]  d_val    - diagonal values
    *
    * @todo Decide how to allow user to configure grid and block sizes.
    */
    void leftScale(index_type n,
                      const index_type* a_row_ptr,
                      real_type* a_val,
                      const real_type* d_val)
    {
      // Define block size and number of blocks
      const int block_size = 1;
      int num_blocks = (n + block_size - 1) / block_size;
      // Launch the kernel
      kernels::leftScale<<<num_blocks, block_size>>>(n, a_row_ptr, a_val, d_val);
    }

    /**
    * @brief Wrapper that scales a csr matrix on the right by a diagonal matrix
    *
    * @param[in]  n      - number of rows in the matrix
    * @param[in]  a_row_ptr - row pointers (CSR storage)
    * @param[in]  a_col_ind - column indices (CSR storage)
    * @param[in, out]  a_val    - values (CSR storage). Changes in place.
    * @param[in]  d_val    - diagonal values
    *
    * @todo Decide how to allow user to configure grid and block sizes.
    */
    void rightScale(index_type n,
                        const index_type* a_row_ptr,
                        const index_type* a_col_ind,
                        real_type* a_val,
                        const real_type* d_val)
    {
      // Define block size and number of blocks
      const int block_size = 256;
      int num_blocks = (n + block_size - 1) / block_size;
      // Launch the kernel
      kernels::rightScale<<<num_blocks, block_size>>>(n, a_row_ptr, a_col_ind, a_val, d_val);
    }
  } // namespace hip
} // namespace ReSolve
