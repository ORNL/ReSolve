#include "RandomizedConjugateGradientCuda.hpp"

#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    namespace kernels
    {
      template <index_type K, index_type BLOCK_SIZE, index_type THREADS_PER_ROW>
      __global__ void SpMMTallSkinnyKernelVector(
        const index_type* __restrict__ A_row_ptr,
        const index_type* __restrict__ A_col_idx,
        const real_type* __restrict__ A_val,
        const real_type* __restrict__ B,
        real_type*  result,
        index_type n)
      {
        const index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
        const index_type row = thread / THREADS_PER_ROW;
        const index_type lane = thread % THREADS_PER_ROW;
        constexpr index_type rows_per_block = BLOCK_SIZE / THREADS_PER_ROW;

        __shared__ real_type result_shared[rows_per_block * K];

        index_type k = K;

        if (row >= n) return;

        // for (index_type row = row_offset; row < n; row += gridDim.x * blockDim.x / 32)
        {
          index_type col_start = A_row_ptr[row];
          index_type col_end   = A_row_ptr[row + 1];

          real_type sum[K] = { 0.0 };

          for (index_type j = col_start + lane; j < col_end; j += THREADS_PER_ROW)
          {
            index_type col = A_col_idx[j];
            real_type a = A_val[j];
            real_type b_local[K];

            // #pragma unroll
            for (int i = 0; i < k; ++i)
            {
              b_local[i] = B[i * n + col];
            }

            for (int i = 0; i < k; ++i)
            {
              sum[i] += a * b_local[i];
            }
          }

          // #pragma unroll
          for (int i = 0; i < k; ++i)
          {
            for (index_type offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2)
            {
              sum[i] += __shfl_down_sync(0xffffffff, sum[i], offset);
            }
          }

          if (lane == 0)
          {
            for (int i = 0; i < k; ++i)
            // result_shared[lane * rows_per_block + threadIdx.x / 32];
              result[i * n + row] = sum[i];
          }

          // if (lane == 0)
          // {
          //   for (int i = 0; i < k; ++i)
          //   {
          //     result_shared[i * rows_per_block + warp] = sum[i];
          //     // result[i * n + row] = sum[i];
          //   }
          // }

          // __syncthreads();
          // if (threadIdx.x < (rows_per_block * k))
          // {
          //   index_type row = threadIdx.x % rows_per_block + blockIdx.x * rows_per_block;
          //   index_type col = threadIdx.x / rows_per_block;
          //   result[col * n + row] = result_shared[threadIdx.x];
          // }
        }
      }
      
      __global__ void columnWiseSquaredNorms(real_type* R, real_type* sq_norms, index_type n)
      {
        index_type col = threadIdx.y;
        index_type row_offset = blockIdx.x * blockDim.x + threadIdx.x;
        index_type stride = gridDim.x * blockDim.x;

        real_type r_sq = 0;
        for (index_type row = row_offset; row < n; row += stride)
        {
          r_sq += R[col * n + row] * R[col * n + row];
        }
        for (index_type offset = 32 / 2; offset > 0; offset /= 2)
        {
          r_sq += __shfl_down_sync(0xffffffff, r_sq, offset);
        }
        if (threadIdx.x == 0)
        {
          atomicAdd(&sq_norms[col], r_sq);
        }
      }

      template <index_type k>
      __global__ void SquaredNormArgMin(real_type* sq_norms, index_type* best_basis)
      {
        index_type thread = threadIdx.x;

        index_type best_basis_local = thread;
        real_type sq_norm_local = sq_norms[thread];

        #pragma unroll
        for (index_type offset = k / 2; offset > 0; offset /= 2)
        {
          constexpr unsigned int REDUCTION_MASK = (1u << k) - 1;
          index_type target_index = __shfl_down_sync(REDUCTION_MASK, best_basis_local, offset);
          real_type target_norm = __shfl_down_sync(REDUCTION_MASK, sq_norm_local, offset);
          if (target_norm < sq_norm_local)
          {
            best_basis_local = target_index;
            sq_norm_local = target_norm;
          }
        }
        __syncwarp();

        if (thread == 0)
        {
          *best_basis = best_basis_local;
          sq_norms[0] = sqrt(sq_norm_local);
        }
      }

      template <index_type k>
      __device__ __forceinline__ index_type indexUpperTriangular(index_type i, index_type j)
      {
        return (j * (j + 1)) / 2 + i;
      }

      template <index_type k>
      __device__ __forceinline__ index_type indexLowerTriangular(index_type i, index_type j)
      {
        return (j * (2 * k - 1 - j)) / 2 + i;
      }

      template <index_type k>
      __global__ void choleskyQr(real_type* W, real_type* R, index_type n)
      {
        // k <= 16
        
        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
        index_type stride = gridDim.x * blockDim.x;

      // Compute Gram matrix
        real_type G_local[k * (k + 1) / 2] = { 0.0 };

        // the sum of n kxk outer products
        for (index_type row = thread; row < n; row += stride)
        {
          real_type w_row[k];

          #pragma unroll
          for (index_type col = 0; col < k; col++)
          {
            w_row[col] = W[col * n + row];
          }

          #pragma unroll
          for (index_type i = 0; i < k; i++)
          {
            #pragma unroll
            for (index_type j = i; j < k; j++) // Only use upper triangle
            {
              G_local[indexUpperTriangular<k>(i, j)] += w_row[i] * w_row[j];
            }
          }
        }

        // Sum reduction to get the sums in each warp
        #pragma unroll
        for (index_type i = 0; i < k; i++)
        {
          #pragma unroll
          for (index_type j = i; j < k; j++)
          {
            real_type g = G_local[indexUpperTriangular<k>(i, j)];

            #pragma unroll
            for (index_type offset = 32 / 2; offset > 0; offset /= 2)
            {
              g += __shfl_down_sync(0xffffffff, g, offset);
            }
            if (threadIdx.x % 32 == 0)
            {
              atomicAdd(&R[j * k + i], g);
            }
          }
        }
        // Now R = W^real_type * W. Lower half is garbage and doesn't matter though, because everything well get overridden soon

        grid.sync();
      
      // Upper Cholesky factorization
        __shared__ real_type R_shared[k * (k + 1) / 2]; // Only need this much for kxk symmetric matrix
        if constexpr (k <= 0) // CAREFUL: SHOULD BE 4?
        {
          if (threadIdx.x < 32)
          {
            index_type i = threadIdx.x % k;
            index_type j = threadIdx.x / k;
            if ((i < k) && (j < k) && (i <= j))
            {
              R_shared[indexUpperTriangular<k>(i, j)] = R[j * k + i];
            }
            __syncwarp();

            // Do all of this inside one warp
            for (index_type h = 0; h < k; h++)
            {
              if (threadIdx.x == 0)
              {
                R_shared[indexUpperTriangular<k>(h, h)] = sqrt(R_shared[indexUpperTriangular<k>(h, h)]);
              }
              __syncwarp();
              if (threadIdx.x > h && threadIdx.x < k)
              {
                R_shared[indexUpperTriangular<k>(h, threadIdx.x)] /= R_shared[indexUpperTriangular<k>(h, h)];
              }
              __syncwarp();
              if ((i < k) && (i > h) && (j < k) && (i <= j))
              {
                R_shared[indexUpperTriangular<k>(i, j)] -= R_shared[indexUpperTriangular<k>(h, i)] * R_shared[indexUpperTriangular<k>(h, j)];
              }
              __syncwarp();
            }
            
            if ((i < k) && (j < k) && (blockIdx.x == 0))
            {
              if (i <= j)
              {
                R[j * k + i] = R_shared[indexUpperTriangular<k>(i, j)];
              }
              else
              {
                R[j * k + i] = 0.0;
              }
            }
          }
          __syncthreads();
        }
        else
        {
          index_type i = threadIdx.x % k;
          index_type j = threadIdx.x / k;
          if ((i < k) && (j < k) && (i <= j))
          {
            R_shared[indexUpperTriangular<k>(i, j)] = R[j * k + i];
          }
          __syncthreads();

          // Do all of this inside one warp
          for (index_type h = 0; h < k; h++)
          {
            if (threadIdx.x == 0)
            {
              R_shared[indexUpperTriangular<k>(h, h)] = sqrt(R_shared[indexUpperTriangular<k>(h, h)]);
            }
            __syncthreads();
            if (threadIdx.x > h && threadIdx.x < k)
            {
              R_shared[indexUpperTriangular<k>(h, threadIdx.x)] /= R_shared[indexUpperTriangular<k>(h, h)];
            }
            __syncthreads();
            if ((i < k) && (i > h) && (j < k) && (i <= j))
            {
              R_shared[indexUpperTriangular<k>(i, j)] -= R_shared[indexUpperTriangular<k>(h, i)] * R_shared[indexUpperTriangular<k>(h, j)];
            }
            __syncthreads();
          }
          
          if ((i < k) && (j < k) && (blockIdx.x == 0))
          {
            if (i <= j)
            {
              R[j * k + i] = R_shared[indexUpperTriangular<k>(i, j)];
            }
            else
            {
              R[j * k + i] = 0.0;
            }
          }
        }
      // No grid-wise sync needed because every block computes R_shared
      
      // Back substitution, QR = A
      // each q is dependent only on q's to the left of it, as well as R and A (W)
        real_type q_row[k];
        for (index_type row = thread; row < n; row += stride)
        {
          #pragma unroll
          for (index_type col = 0; col < k; col++)
          {
            real_type q_local = W[col * n + row];
            #pragma unroll
            for (index_type i = 0; i < col; i++)
            {
              q_local -= q_row[i] * R_shared[indexUpperTriangular<k>(i, col)];
            }
            q_local /= R_shared[indexUpperTriangular<k>(col, col)];
            q_row[col] = q_local;
          }

          #pragma unroll
          for (index_type col = 0; col < k; col++)
          {
            W[col * n + row] = q_row[col];
          }
        }
      }

      template <index_type k>
      __global__ void updateXRSplit(real_type* __restrict__ Xi_inv,
                               const real_type* __restrict__ Sigma,
                               const real_type* __restrict__ S,
                               const real_type* __restrict__ A_S,
                               real_type* __restrict__ Xi_Sigma,
                               real_type* __restrict__ X_res,
                               real_type* __restrict__ R_prec,
                               index_type n)
      {
        index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
        index_type stride = gridDim.x * blockDim.x;

        __shared__ real_type Xi_Sigma_shared[k * k];
        if (threadIdx.x < k * k)
        {
          Xi_Sigma_shared[threadIdx.x] = Xi_Sigma[threadIdx.x];
        }
        __syncthreads();

        for (index_type row = thread; row < n; row += gridDim.x * blockDim.x)
        {
          // #pragma unroll 1
          for (index_type col = 0; col < k; col++)
          {
            real_type x_dot = 0;
            real_type r_dot = 0;
            // #pragma unroll 1
            for (index_type i = 0; i < k; i++)
            {
              x_dot +=   S[i * n + row] * Xi_Sigma_shared[col * k + i];
              r_dot -= A_S[i * n + row] * Xi_Sigma_shared[col * k + i];
            }
            X_res[col * n + row] += x_dot;
            R_prec[col * n + row] += r_dot;
          }
        }
      }
      
      template <index_type k>
      __global__ void choleskyFactorizeSolve(real_type* __restrict__ A,
                               const real_type* B,
                               real_type* X)
      {
        index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
        index_type stride = gridDim.x * blockDim.x;
      
      // choleskyFactorize(A), lower
        __shared__ real_type A_shared[k * (k + 1) / 2]; // Only need this much for kxk symmetric matrix
        if constexpr (k <= 4)
        {
          if (threadIdx.x < 32) // k <= 4 for now
          {
            index_type i = threadIdx.x % k;
            index_type j = threadIdx.x / k;
            if ((i < k) && (j < k) && (i >= j))
            {
              A_shared[indexLowerTriangular<k>(i, j)] = A[j * k + i];
            }
            __syncwarp();

            // // For warp size 64 and k <= 8, everything can be done in one warp
            // if constexpr (k * (k + 1) / 2 > 64)
            // {
            //   __syncthreads();
            // }

            // Do all of this inside one warp. One thread per entry
            // #pragma unroll 1
            for (index_type h = 0; h < k; h++)
            {
              if (threadIdx.x == 0)
              {
                A_shared[indexLowerTriangular<k>(h, h)] = sqrt(A_shared[indexLowerTriangular<k>(h, h)]);
              }
              __syncwarp();
              if (threadIdx.x > h && threadIdx.x < k)
              {
                A_shared[indexLowerTriangular<k>(threadIdx.x, h)] /= A_shared[indexLowerTriangular<k>(h, h)];
              }
              __syncwarp();
              if ((j < k) && (j > h) && (i < k) && (i >= j))
              {
                A_shared[indexLowerTriangular<k>(i, j)] -= A_shared[indexLowerTriangular<k>(i, h)] * A_shared[indexLowerTriangular<k>(j, h)];
              }
              __syncwarp();
            }
            
            if ((i < k) && (j < k) && (blockIdx.x == 0))
            {
              if (i >= j)
              {
                A[j * k + i] = A_shared[indexLowerTriangular<k>(i, j)];
              }
              else
              {
                A[j * k + i] = 0.0;
              }
            }
          }
        }
        else
        {
          index_type i = threadIdx.x % k;
          index_type j = threadIdx.x / k;
          if ((i < k) && (j < k) && (i >= j))
          {
            A_shared[indexLowerTriangular<k>(i, j)] = A[j * k + i];
          }
          __syncthreads();

          // #pragma unroll 1
          for (index_type h = 0; h < k; h++)
          {
            if (threadIdx.x == 0)
            {
              A_shared[indexLowerTriangular<k>(h, h)] = sqrt(A_shared[indexLowerTriangular<k>(h, h)]);
            }
            __syncthreads();
            if (threadIdx.x > h && threadIdx.x < k)
            {
              A_shared[indexLowerTriangular<k>(threadIdx.x, h)] /= A_shared[indexLowerTriangular<k>(h, h)];
            }
            __syncthreads();
            if ((j < k) && (j > h) && (i < k) && (i >= j))
            {
              A_shared[indexLowerTriangular<k>(i, j)] -= A_shared[indexLowerTriangular<k>(i, h)] * A_shared[indexLowerTriangular<k>(j, h)];
            }
            __syncthreads();
          }
          __syncthreads();
          
          if ((i < k) && (j < k) && (blockIdx.x == 0))
          {
            if (i >= j)
            {
              A[j * k + i] = A_shared[indexLowerTriangular<k>(i, j)];
            }
            else
            {
              A[j * k + i] = 0.0;
            }
          }
        }

      // X = Xi * B => A * X = B => L * L^real_type * X = B, choleskySolve
        __shared__ real_type Xi_Sigma_shared[k * k];
        if constexpr (k <= 4)
        {
          if (threadIdx.x < 32) // k <= 4 for now. Still only one warp
          {
            if (threadIdx.x < k * k)
            {
              Xi_Sigma_shared[threadIdx.x] = B[threadIdx.x];
            }
            __syncwarp(); // CUDA needs to __syncwarp() at all these places

            //  L * Y = B
            if (threadIdx.x < k)
            {
              index_type col = threadIdx.x; // Talking about rows and columns of B

              // #pragma unroll 1
              for (index_type row = 0; row < k; row++)
              {
                real_type y_local = Xi_Sigma_shared[col * k + row];
                // #pragma unroll 1
                for (index_type i = 0; i < row; i++)
                {
                  y_local -= Xi_Sigma_shared[col * k + i] * A_shared[indexLowerTriangular<k>(row, i)]; // A_shared can be overridden. Here it contains Y
                }
                y_local /= A_shared[indexLowerTriangular<k>(row, row)];
                Xi_Sigma_shared[col * k + row] = y_local;
              }
            }
            __syncwarp();

            //  L^real_type * X = Y
            if (threadIdx.x < k)
            {
              index_type col = threadIdx.x; // Talking about rows and columns of B

              // #pragma unroll 1
              for (index_type row = k - 1; row >= 0; row--)
              {
                real_type y_local = Xi_Sigma_shared[col * k + row];
                // #pragma unroll 1
                for (index_type i = row + 1; i < k; i++)
                {
                  y_local -= Xi_Sigma_shared[col * k + i] * A_shared[indexLowerTriangular<k>(i, row)]; // A_shared can be overridden. Here it contains Y
                }
                y_local /= A_shared[indexLowerTriangular<k>(row, row)];
                Xi_Sigma_shared[col * k + row] = y_local;
              }
            }
          }
        }
        else
        {
          if (threadIdx.x < k * k)
          {
            Xi_Sigma_shared[threadIdx.x] = B[threadIdx.x];
          }
          __syncthreads();

          //  L * Y = B
          if (threadIdx.x < k)
          {
            index_type col = threadIdx.x; // Talking about rows and columns of B

            // #pragma unroll 1
            for (index_type row = 0; row < k; row++)
            {
              real_type y_local = Xi_Sigma_shared[col * k + row];
              // #pragma unroll 1
              for (index_type i = 0; i < row; i++)
              {
                y_local -= Xi_Sigma_shared[col * k + i] * A_shared[indexLowerTriangular<k>(row, i)]; // A_shared can be overridden. Here it contains Y
              }
              y_local /= A_shared[indexLowerTriangular<k>(row, row)];
              Xi_Sigma_shared[col * k + row] = y_local;
            }
          }
          __syncthreads();

          //  L^real_type * X = Y
          if (threadIdx.x < k)
          {
            index_type col = threadIdx.x; // Talking about rows and columns of B

            // #pragma unroll 1
            for (index_type row = k - 1; row >= 0; row--)
            {
              real_type y_local = Xi_Sigma_shared[col * k + row];
              // #pragma unroll 1
              for (index_type i = row + 1; i < k; i++)
              {
                y_local -= Xi_Sigma_shared[col * k + i] * A_shared[indexLowerTriangular<k>(i, row)]; // A_shared can be overridden. Here it contains Y
              }
              y_local /= A_shared[indexLowerTriangular<k>(row, row)];
              Xi_Sigma_shared[col * k + row] = y_local;
            }
          }
        }
        __syncthreads();
        
        if (threadIdx.x < k * k)
        {
          X[threadIdx.x] = Xi_Sigma_shared[threadIdx.x];
        }
      }

      // W = W - B * L^-1
      template <index_type k>
      __global__ void updateW(real_type* __restrict__ W, const real_type* __restrict__ L, real_type* __restrict__ B, index_type n)
      {
        index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
        index_type stride = gridDim.x * blockDim.x;
        
        __shared__ real_type L_shared[k * (k + 1) / 2]; // Only need this much for kxk symmetric matrix

        {
          index_type i = threadIdx.x % k;
          index_type j = threadIdx.x / k;
          if ((i < k) && (j < k) && (i >= j))
          {
            L_shared[indexLowerTriangular<k>(i, j)] = L[j * k + i];
          }
        }
        __syncthreads();

        for (index_type row = thread; row < n; row += stride)
        {
          //  Y * L^T = B
          real_type y_row[k];

          #pragma unroll
          for (index_type col = 0; col < k; col++)
          {
            real_type y_local = B[col * n + row];
            #pragma unroll
            for (index_type i = 0; i < col; i++)
            {
              y_local -= y_row[i] * L_shared[indexLowerTriangular<k>(col, i)]; // L^T[i, col]
            }
            y_local /= L_shared[indexLowerTriangular<k>(col, col)];
            y_row[col] = y_local;
          }

          #pragma unroll
          for (index_type col = 0; col < k; col++)
          {
            B[col * n + row] = y_row[col]; // Y is stored in B. B is overridden
          }
        
          // X * L = Y
          real_type x_row[k];

          #pragma unroll
          for (index_type col = k - 1; col >= 0; col--)
          {
            real_type x_local = B[col * n + row];
            #pragma unroll
            for (index_type i = col + 1; i < k; i++)
            {
              x_local -= x_row[i] * L_shared[indexLowerTriangular<k>(i, col)];
            }
            x_local /= L_shared[indexLowerTriangular<k>(col, col)];
            x_row[col] = x_local;
          }

          #pragma unroll
          for (index_type col = 0; col < k; col++)
          {
            W[col * n + row] -= x_row[col];
          }
        }
      }

      template <index_type k>
      __global__ void multTSMTTSM(const real_type* A, const real_type* B, real_type* __restrict__ C, index_type n)
      {
        index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
        index_type stride = gridDim.x * blockDim.x;

        __shared__ real_type C_shared[k * k];
        if (threadIdx.x < k * k)
        {
          C_shared[threadIdx.x] = 0.0;
        }
        __syncthreads();

        // the sum of n kxk outer products
        for (index_type row = thread; row < n; row += stride)
        {
          real_type a_row[k];
          real_type b_row[k];

          #pragma unroll
          for (index_type col = 0; col < k; col++)
          {
            a_row[col] = A[col * n + row];
            b_row[col] = B[col * n + row];
          }

          #pragma unroll
          for (index_type i = threadIdx.x; i < threadIdx.x + k * k; i++)
          {
            index_type idx = i % (k * k);
            atomicAdd(&C_shared[idx], a_row[idx % k] * b_row[idx / k]);
          }
        }
        __syncthreads();
        
        if (threadIdx.x < k * k)
        {
          atomicAdd(&C[threadIdx.x], C_shared[threadIdx.x]);
        }
      }

      template <index_type k, index_type BLOCK_DIM>
      __global__ void updateSSigma(const real_type* __restrict__ W, real_type* __restrict__ S, const real_type* __restrict__ Zeta, real_type* __restrict__ Sigma, index_type n)
      {
        index_type thread = blockIdx.x * BLOCK_DIM + threadIdx.x;

        __shared__ real_type Zeta_shared[k * k];
        if (threadIdx.x < k * k)
        {
          Zeta_shared[threadIdx.x] = Zeta[threadIdx.x];
        }
      
        __shared__ real_type S_shared[BLOCK_DIM * k];
        __shared__ real_type result_shared[BLOCK_DIM * k];
        // Compute S = W + P * Zeta^real_type
        if (blockIdx.x < (n + BLOCK_DIM - 1) / BLOCK_DIM)
        {
          // Cache a horizontally-sliced block of S for coalesced reads
          for (index_type i = threadIdx.x; i < BLOCK_DIM * k; i += BLOCK_DIM)
          {
            index_type block_row = i % BLOCK_DIM;
            index_type global_row = block_row + blockIdx.x * BLOCK_DIM;
            index_type col = i / BLOCK_DIM;
            if (global_row < n)
            {
              S_shared[i] = S[col * n + global_row];
            } else {
              S_shared[i] = 0.0;
            }
            result_shared[i] = 0.0;
          }
          __syncthreads();

          if (thread < n)
          {
            
            for (index_type j = 0; j < k; j++)
            {
              real_type dot = 0.0;
              for (index_type i = 0; i < k; i++)
              {
                dot += S_shared[i * BLOCK_DIM + threadIdx.x] * Zeta_shared[i * k + j]; // Remember, Zeta is transposed
              }
              result_shared[j * BLOCK_DIM + threadIdx.x] = dot;
            }
          }
          __syncthreads();

          #pragma unroll
          for (index_type i = threadIdx.x; i < BLOCK_DIM * k; i += BLOCK_DIM)
          {
            index_type block_row = i % BLOCK_DIM;
            index_type global_row = block_row + blockIdx.x * BLOCK_DIM;
            index_type col = i / BLOCK_DIM;
            if (global_row < n)
            {
              S[col * n + global_row] = W[col * n + global_row] + result_shared[i];
            }
            result_shared[i] = 0.0;
          }
        }

        //////////////////////////////////

        // Compute Sigma = Zeta * Sigma. Block size must be at least k^2
        if (thread < k * k)
        {
          S_shared[thread] = Sigma[thread];
        }
        __syncthreads();
        
        // Each thread takes one entry of the result matrix
        if (thread < k * k)
        {
          index_type row = thread % k;
          index_type col = thread / k;

          real_type dot = 0.0;
          #pragma unroll
          for (index_type i = 0; i < k; i++)
          {
            dot += Zeta_shared[row + i * k] * S_shared[col * k + i];
          }
          result_shared[col * k + row] = dot;
        }
        __syncthreads();

        if (thread < k * k)
        {
          Sigma[thread] = result_shared[thread];
        }
      }

      __global__ void preconditionDense(real_type* __restrict__ A, const real_type* __restrict__ d, index_type n)
      {
        int thread = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (thread < n * n)
        {
          int row = thread % n;
          int col = thread / n;
          A[thread] *= (d[row] * d[col]);
        }
      }
    } // namespace kernels
    
    RandomizedConjugateGradientCuda::RandomizedConjugateGradientCuda(VectorHandler* vector_handler)
    {
      vector_handler_ = vector_handler;

      int device_id = 0;
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, device_id);
      num_sms_     = properties.multiProcessorCount; // 80
      num_threads_ = num_sms_ * properties.maxThreadsPerMultiProcessor; // 163840
      printf("Total SMs: %d, total threads: %d\n", num_sms_, num_threads_);
    }

    RandomizedConjugateGradientCuda::~RandomizedConjugateGradientCuda()
    {
      mem_.deleteOnDevice(d_best_basis_);
      mem_.deleteOnDevice(d_sq_norms_);
    }

    int RandomizedConjugateGradientCuda::setup(index_type k)
    {
      // For choleskyQr()
      switch (k)
      {
      case 1: 
        cholesky_qr_kernel_ = kernels::choleskyQr<1>; 
        break;
      case 2: 
        cholesky_qr_kernel_ = kernels::choleskyQr<2>; 
        break;
      case 4: 
        cholesky_qr_kernel_ = kernels::choleskyQr<4>; 
        break;
      case 8: 
        cholesky_qr_kernel_ = kernels::choleskyQr<8>; 
        break;
      case 16:
        cholesky_qr_kernel_ = kernels::choleskyQr<16>;
        break;
      default:
        return 1;
      }
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&qr_blocks_per_sm_,
                                                    cholesky_qr_kernel_,
                                                    256,
                                                    0);

      // For bestBasis()
      if (!d_best_basis_)
      {
        mem_.allocateArrayOnDevice(&d_best_basis_, 1);
      }
      if (d_sq_norms_)
      {
        mem_.deleteOnDevice(d_sq_norms_);
        d_sq_norms_ = nullptr;
      }
      mem_.allocateArrayOnDevice(&d_sq_norms_, k);

      return 0;
    }

    int RandomizedConjugateGradientCuda::SpMMTallSkinny(matrix::Csr* A, vector::Vector* X, vector::Vector* result)
    {
      index_type n = A->getNumRows();
      index_type k = X->getNumVectors();

      constexpr int       block_size = 256;
      int       num_blocks = (n * 32 + block_size - 1) / block_size;

      
      #define SPMM_LAUNCH(THREADS_PER_ROW) \
        switch (k) \
        { \
        case 1: \
          kernels::SpMMTallSkinnyKernelVector<1, block_size, THREADS_PER_ROW><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE), \
                                                                  A->getColData(memory::DEVICE), \
                                                                  A->getValues(memory::DEVICE), \
                                                                  X->getData(memory::DEVICE), \
                                                                  result->getData(memory::DEVICE), \
                                                                  n); \
          break; \
        case 2: \
          kernels::SpMMTallSkinnyKernelVector<2, block_size, THREADS_PER_ROW><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE), \
                                                                  A->getColData(memory::DEVICE), \
                                                                  A->getValues(memory::DEVICE), \
                                                                  X->getData(memory::DEVICE), \
                                                                  result->getData(memory::DEVICE), \
                                                                  n); \
          break; \
        case 4: \
          kernels::SpMMTallSkinnyKernelVector<4, block_size, THREADS_PER_ROW><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE), \
                                                                  A->getColData(memory::DEVICE), \
                                                                  A->getValues(memory::DEVICE), \
                                                                  X->getData(memory::DEVICE), \
                                                                  result->getData(memory::DEVICE), \
                                                                  n); \
          break; \
        case 8: \
          kernels::SpMMTallSkinnyKernelVector<8, block_size, THREADS_PER_ROW><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE), \
                                                                  A->getColData(memory::DEVICE), \
                                                                  A->getValues(memory::DEVICE), \
                                                                  X->getData(memory::DEVICE), \
                                                                  result->getData(memory::DEVICE), \
                                                                  n); \
          break; \
        case 16: \
          kernels::SpMMTallSkinnyKernelVector<16, block_size, THREADS_PER_ROW><<<num_blocks, block_size>>>(A->getRowData(memory::DEVICE), \
                                                                  A->getColData(memory::DEVICE), \
                                                                  A->getValues(memory::DEVICE), \
                                                                  X->getData(memory::DEVICE), \
                                                                  result->getData(memory::DEVICE), \
                                                                  n); \
          break; \
        default: \
          return 1; \
        }

      int nnz_ratio = A->getNnz() / n;
      if (nnz_ratio >= 32)
      {
        SPMM_LAUNCH(32);
      }
      else if (nnz_ratio >= 5) // up to 25
      {
        SPMM_LAUNCH(8);
      }
      else
      {
        SPMM_LAUNCH(2);
      }
      
      return 0;
    }
    
    int RandomizedConjugateGradientCuda::bestBasis(vector::Vector* R, index_type* h_best_basis, real_type* h_best_basis_norm)
    {
      index_type n = R->getSize();
      index_type k = R->getNumVectors();

      dim3       block_size(32, k);
      int        num_blocks = num_sms_ * 4; // Pretty arbitrary. Tune this later
      cudaMemsetAsync(d_sq_norms_, 0.0, k * sizeof(real_type));
      kernels::columnWiseSquaredNorms<<<num_blocks, block_size>>>(R->getData(memory::DEVICE), d_sq_norms_, n);

      // Kernel writes back into first entry of d_sq_norms_
      switch (k)
      {
      case 1:
        kernels::SquaredNormArgMin<1><<<1, k>>>(d_sq_norms_, d_best_basis_);
        break;
      case 2:
        kernels::SquaredNormArgMin<2><<<1, k>>>(d_sq_norms_, d_best_basis_);
        break;
      case 4:
        kernels::SquaredNormArgMin<4><<<1, k>>>(d_sq_norms_, d_best_basis_);
        break;
      case 8:
        kernels::SquaredNormArgMin<8><<<1, k>>>(d_sq_norms_, d_best_basis_);
        break;
      case 16:
        kernels::SquaredNormArgMin<16><<<1, k>>>(d_sq_norms_, d_best_basis_);
        break;
      default:
        return 1;
      }

      cudaMemcpyAsync(h_best_basis, d_best_basis_, sizeof(index_type), cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(h_best_basis_norm, d_sq_norms_, sizeof(real_type), cudaMemcpyDeviceToHost);

      return 0;
    }

    // R must be zeroed
    int RandomizedConjugateGradientCuda::choleskyQr(vector::Vector* W, vector::Vector* R, memory::MemorySpace memspace)
    {
      index_type n = W->getSize();
      index_type k = W->getNumVectors();

      if (k == 1)
      {
        real_type W_norm = vector_handler_->norm(W, memory::DEVICE);
        vector_handler_->scal(1 / W_norm, W, memory::DEVICE);
        R->setToConst(W_norm, memory::DEVICE);
        return 0;
      }

      cudaError_t status;

      real_type* d_W = W->getData(memory::DEVICE);
      real_type* d_R = R->getData(memory::DEVICE);
      
      void* args[] = {
          (void*)&d_W,
          (void*)&d_R,
          (void*)&n
      };
      
      int       block_size = 256; // Must be at least k^2
      int       num_blocks = num_sms_ * qr_blocks_per_sm_;

      status = cudaLaunchCooperativeKernel((void*)cholesky_qr_kernel_, num_blocks, block_size, args, 0, 0);
      
      if (status != cudaSuccess) {
          return 1;
      }

      return 0;
    }
    
    int RandomizedConjugateGradientCuda::updateXRSplit(vector::Vector* Xi_inv, vector::Vector* Sigma, vector::Vector* S, vector::Vector* A_S, vector::Vector* Xi_Sigma, vector::Vector* X_res, vector::Vector* R_prec)
    {
      index_type n = A_S->getSize();
      index_type k = A_S->getNumVectors();

      int       block_size = (k <= 4) ? 32 : 64; // Must be at least k^2
      int       num_blocks = 1; // quite arbitrary

      switch (k)
      {
      case 1:
        kernels::choleskyFactorizeSolve<1><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE));
        break;
      case 2:
        kernels::choleskyFactorizeSolve<2><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE));
        break;
      case 4:
        kernels::choleskyFactorizeSolve<4><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE));
        break;
      case 8:
        kernels::choleskyFactorizeSolve<8><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE));
        break;
      case 16:
        kernels::choleskyFactorizeSolve<16><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE));
        break;
      default:
        return 1;
      }

      block_size = 256; // Must be at least k^2
      num_blocks = num_sms_ * 64; // quite arbitrary

      switch (k)
      {
      case 1:
        kernels::updateXRSplit<1><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      case 2:
        kernels::updateXRSplit<2><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      case 4:
        kernels::updateXRSplit<4><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      case 8:
        kernels::updateXRSplit<8><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      case 16:
        kernels::updateXRSplit<16><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      default:
        return 1;
      }

      return 0;
    }

    int RandomizedConjugateGradientCuda::choleskyFactorizeSolve(vector::Vector* A, vector::Vector* B, vector::Vector* X)
    {
      index_type k = A->getSize();

      int       block_size = (k <= 4) ? 32 : 64; // Must be at least k^2
      int       num_blocks = 1; // quite arbitrary

      switch (k)
      {
      case 1:
        kernels::choleskyFactorizeSolve<1><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                         B->getData(memory::DEVICE),
                                                         X->getData(memory::DEVICE));
        break;
      case 2:
        kernels::choleskyFactorizeSolve<2><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                         B->getData(memory::DEVICE),
                                                         X->getData(memory::DEVICE));
        break;
      case 4:
        kernels::choleskyFactorizeSolve<4><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                         B->getData(memory::DEVICE),
                                                         X->getData(memory::DEVICE));
        break;
      case 8:
        kernels::choleskyFactorizeSolve<8><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                         B->getData(memory::DEVICE),
                                                         X->getData(memory::DEVICE));
        break;
      case 16:
        kernels::choleskyFactorizeSolve<16><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                         B->getData(memory::DEVICE),
                                                         X->getData(memory::DEVICE));
        break;
      default:
        return 1;
      }

      return 0;
    }

    int RandomizedConjugateGradientCuda::updateW(vector::Vector* W, vector::Vector* L, vector::Vector* B, memory::MemorySpace memspace)
    {
      index_type n = B->getSize();
      index_type k = B->getNumVectors();

      int       block_size = 256; // Must be at least k^2
      int       num_blocks = num_sms_ * 32; // quite arbitrary

      switch (k)
      {
      case 1:
        kernels::updateW<1><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   L->getData(memory::DEVICE),
                                                                   B->getData(memory::DEVICE),
                                                                   n);
        break;
      case 2:
        kernels::updateW<2><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   L->getData(memory::DEVICE),
                                                                   B->getData(memory::DEVICE),
                                                                   n);
        break;
      case 4:
        kernels::updateW<4><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   L->getData(memory::DEVICE),
                                                                   B->getData(memory::DEVICE),
                                                                   n);
        break;
      case 8:
        kernels::updateW<8><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   L->getData(memory::DEVICE),
                                                                   B->getData(memory::DEVICE),
                                                                   n);
        break;
      case 16:
        kernels::updateW<16><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   L->getData(memory::DEVICE),
                                                                   B->getData(memory::DEVICE),
                                                                   n);
        break;
      default:
        return 1;
      }

      return 0;
    }

    // C = A^real_type * B
    int RandomizedConjugateGradientCuda::multTSMTTSM(vector::Vector* A, vector::Vector* B, vector::Vector* C, memory::MemorySpace memspace)
    {
      index_type n = A->getSize();
      index_type k = A->getNumVectors();

      int       block_size = 256;
      int       num_blocks = num_sms_ * 32;
      
      C->setToZero(memory::DEVICE);
      switch (k)
      {
      case 1:
        kernels::multTSMTTSM<1><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                            B->getData(memory::DEVICE),
                                                            C->getData(memory::DEVICE),
                                                            n);
        break;
      case 2:
        kernels::multTSMTTSM<2><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                            B->getData(memory::DEVICE),
                                                            C->getData(memory::DEVICE),
                                                            n);
        break;
      case 4:
        kernels::multTSMTTSM<4><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                            B->getData(memory::DEVICE),
                                                            C->getData(memory::DEVICE),
                                                            n);
        break;
      case 8:
        kernels::multTSMTTSM<8><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                            B->getData(memory::DEVICE),
                                                            C->getData(memory::DEVICE),
                                                            n);
        break;
      case 16:
        kernels::multTSMTTSM<16><<<num_blocks, block_size>>>(A->getData(memory::DEVICE),
                                                            B->getData(memory::DEVICE),
                                                            C->getData(memory::DEVICE),
                                                            n);
        break;
      default:
        return 1;
      }

      return 0;
    }

    int RandomizedConjugateGradientCuda::updateSSigma(vector::Vector* W, vector::Vector* S, vector::Vector* Zeta, vector::Vector* Sigma, memory::MemorySpace memspace)
    {
      index_type n = W->getSize();
      index_type k = W->getNumVectors();  

      // One thread per row for S = W + S * Zeta^real_type
      // One block for Sigma = Zeta * Sigma
      constexpr int block_size = 128; // Must be at least k^2. IMPROVE UPON THIS??
      int           num_blocks = (n + block_size - 1) / block_size + 1;

      switch (k)
      {
      case 1:
        kernels::updateSSigma<1, block_size><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   S->getData(memory::DEVICE),
                                                                   Zeta->getData(memory::DEVICE),
                                                                   Sigma->getData(memory::DEVICE),
                                                                   n);
        break;
      case 2:
        kernels::updateSSigma<2, block_size><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   S->getData(memory::DEVICE),
                                                                   Zeta->getData(memory::DEVICE),
                                                                   Sigma->getData(memory::DEVICE),
                                                                   n);
        break;
      case 4:
        kernels::updateSSigma<4, block_size><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   S->getData(memory::DEVICE),
                                                                   Zeta->getData(memory::DEVICE),
                                                                   Sigma->getData(memory::DEVICE),
                                                                   n);
        break;
      case 8:
        kernels::updateSSigma<8, block_size><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   S->getData(memory::DEVICE),
                                                                   Zeta->getData(memory::DEVICE),
                                                                   Sigma->getData(memory::DEVICE),
                                                                   n);
        break;
      case 16:
        kernels::updateSSigma<16, block_size><<<num_blocks, block_size>>>(W->getData(memory::DEVICE),
                                                                   S->getData(memory::DEVICE),
                                                                   Zeta->getData(memory::DEVICE),
                                                                   Sigma->getData(memory::DEVICE),
                                                                   n);
        break;
      default:
        return 1;
      }

      return 0;
    }

    int RandomizedConjugateGradientCuda::preconditionDense(vector::Vector* A, vector::Vector* d)
    {
      index_type n = A->getSize();

      constexpr int block_size = 256;
      int num_blocks = (n * n + block_size - 1) / block_size;
      kernels::preconditionDense<<<num_blocks, block_size>>>(A->getData(memory::DEVICE), d->getData(memory::DEVICE), n);

      return 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /******************************************************************************
    * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
    * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
    *
    * SPDX-License-Identifier: (Apache-2.0 OR MIT)
    ******************************************************************************/
    using hypre_DeviceItem = void*;
    #ifndef hypre_CSR_SPMV_DEVICE_H
    #define hypre_CSR_SPMV_DEVICE_H

    #define HYPRE_SPMV_BLOCKDIM 512
    #define HYPRE_SPMV_VERSION 1
    #define HYPRE_SPMV_FILL_STRICT_LOWER -2
    #define HYPRE_SPMV_FILL_LOWER -1
    #define HYPRE_SPMV_FILL_WHOLE 0
    #define HYPRE_SPMV_FILL_UPPER 1
    #define HYPRE_SPMV_FILL_STRICT_UPPER 2

    #define HYPRE_SPMV_ADD_SUM(p, nv)                                                         \
    {                                                                                         \
      const index_type col = __ldg(&d_ja[p]);                                        \
      if constexpr (F == HYPRE_SPMV_FILL_WHOLE)                                                        \
      {                                                                                      \
          const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                                 \
          for (index_type i = 0; i < nv; i++)                                                  \
          {                                                                                   \
            sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);       \
          }                                                                                   \
      }                                                                                      \
      else if constexpr (F == HYPRE_SPMV_FILL_LOWER)                                                   \
      {                                                                                      \
          if (col <= grid_group_id)                                                           \
          {                                                                                   \
            const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                              \
            for (index_type i = 0; i < nv; i++)                                               \
            {                                                                                \
                sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);    \
            }                                                                                \
          }                                                                                   \
      }                                                                                      \
      else if constexpr (F == HYPRE_SPMV_FILL_UPPER)                                                   \
      {                                                                                      \
          if (col >= grid_group_id)                                                           \
          {                                                                                   \
            const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                              \
            for (index_type i = 0; i < nv; i++)                                               \
            {                                                                                \
                sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);    \
            }                                                                                \
          }                                                                                   \
      }                                                                                      \
      else if constexpr (F == HYPRE_SPMV_FILL_STRICT_LOWER)                                            \
      {                                                                                      \
          if (col < grid_group_id)                                                            \
          {                                                                                   \
            const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                              \
            for (index_type i = 0; i < nv; i++)                                               \
            {                                                                                \
                sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);    \
            }                                                                                \
          }                                                                                   \
      }                                                                                      \
      else if constexpr (F == HYPRE_SPMV_FILL_STRICT_UPPER)                                            \
      {                                                                                      \
          if (col > grid_group_id)                                                            \
          {                                                                                   \
            const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                              \
            for (index_type i = 0; i < nv; i++)                                               \
            {                                                                                \
                sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);    \
            }                                                                                \
          }                                                                                   \
      }                                                                                      \
    }

    #define HYPRE_SPMV_GPU_LAUNCH(kernel, nv)                                                  \
      if (avg_rownnz >= avg_rownnz_lower_bounds[0])                                           \
      {                                                                                       \
          const dim3 gDim =                                                                    \
            dim3((num_rows + num_groups_per_block[0] - 1) / num_groups_per_block[0]);   \
          HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[0], nv, real_type>),                                \
                            gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,             \
                            idxstride_y, vecstride_x, vecstride_y, alpha,                      \
                            d_ia, d_ja, d_a, d_x, beta, d_y );                                 \
      }                                                                                       \
      else if (avg_rownnz >= avg_rownnz_lower_bounds[1])                                      \
      {                                                                                       \
          const dim3 gDim =                                                                    \
            dim3((num_rows + num_groups_per_block[1] - 1) / num_groups_per_block[1]);   \
          HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[1], nv, real_type>),                                \
                            gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,             \
                            idxstride_y, vecstride_x, vecstride_y, alpha,                      \
                            d_ia, d_ja, d_a, d_x, beta, d_y );                                 \
      }                                                                                       \
      else if (avg_rownnz >= avg_rownnz_lower_bounds[2])                                      \
      {                                                                                       \
          const dim3 gDim =                                                                    \
            dim3((num_rows + num_groups_per_block[2] - 1) / num_groups_per_block[2]);   \
          HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[2], nv, real_type>),                                \
                            gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,             \
                            idxstride_y, vecstride_x, vecstride_y, alpha,                      \
                            d_ia, d_ja, d_a, d_x, beta, d_y );                                 \
      }                                                                                       \
      else if (avg_rownnz >= avg_rownnz_lower_bounds[3])                                      \
      {                                                                                       \
          const dim3 gDim =                                                                    \
            dim3((num_rows + num_groups_per_block[3] - 1) / num_groups_per_block[3]);   \
          HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[3], nv, real_type>),                                \
                            gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,             \
                            idxstride_y, vecstride_x, vecstride_y, alpha,                      \
                            d_ia, d_ja, d_a, d_x, beta, d_y );                                 \
      }                                                                                       \
      else                                                                                    \
      {                                                                                       \
          const dim3 gDim =                                                                    \
            dim3((num_rows + num_groups_per_block[4] - 1) / num_groups_per_block[4]);   \
          HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[4], nv, real_type>),                                \
                            gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,             \
                            idxstride_y, vecstride_x, vecstride_y, alpha,                      \
                            d_ia, d_ja, d_a, d_x, beta, d_y );                                 \
      }

      #define HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) kernel_name<<<gridsize, blocksize>>>(__VA_ARGS__)

    #endif

        
    template <index_type F, index_type K, index_type NV, typename T>
    __global__ void
    hypreGPUKernel_CSRMatvecShuffleGT8(hypre_DeviceItem &item,
                                      index_type         num_rows,
                                      index_type         num_vectors,
                                      index_type        *row_id,
                                      index_type         idxstride_x,
                                      index_type         idxstride_y,
                                      index_type         vecstride_x,
                                      index_type         vecstride_y,
                                      T                 alpha,
                                      index_type        *d_ia,
                                      index_type        *d_ja,
                                      T                *d_a,
                                      T                *d_x,
                                      T                 beta,
                                      T                *d_y )
    {
    #if defined (HYPRE_USING_SYCL)
      const index_type  grid_ngroups  = item.get_group_range(2) * (HYPRE_SPMV_BLOCKDIM / K);
      index_type        grid_group_id = __shfl_sync(et_group(2) * HYPRE_SPMV_BLOCKDIM + item.get_local_id(
                                            2)) / K;
      const index_type  group_lane    = item.get_local_id(2) & (K - 1);
    #else
      const index_type  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
      index_type        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
      const index_type  group_lane    = threadIdx.x & (K - 1);
    #endif
      T sum[64];

      for (; __any_sync(0xffffffff, grid_group_id < num_rows);
            grid_group_id += grid_ngroups)
      {
          index_type grid_row_id = -1, p = 0, q = 0;

          if (row_id)
          {
            if (grid_group_id < num_rows && group_lane == 0)
            {
                grid_row_id = __ldg(&row_id[grid_group_id]);
            }
            grid_row_id = __shfl_sync(0xffffffff, grid_row_id, 0, K);
          }
          else
          {
            grid_row_id = grid_group_id;
          }

          if (grid_group_id < num_rows && group_lane < 2)
          {
            p = __ldg(&d_ia[grid_row_id + group_lane]);
          }
          q = __shfl_sync(0xffffffff, p, 1, K);
          p = __shfl_sync(0xffffffff, p, 0, K);

          for (index_type i = 0; i < num_vectors; i++)
          {
            sum[i] = T(0.0);
          }

    #pragma unroll 1
          for (p += group_lane; p < q; p += K * 2)
          {
            HYPRE_SPMV_ADD_SUM(p, num_vectors)
            if (p + K < q)
            {
                HYPRE_SPMV_ADD_SUM((p + K), num_vectors)
            }
          }

          // parallel reduction
          for (index_type i = 0; i < num_vectors; i++)
          {
            for (index_type d = K / 2; d > 0; d >>= 1)
            {
                sum[i] += __shfl_down_sync(0xffffffff, sum[i], d);
            }
          }

          if (grid_group_id < num_rows && group_lane == 0)
          {
            if (beta)
            {
                for (index_type i = 0; i < num_vectors; i++)
                {
                  d_y[grid_row_id * idxstride_y + i * vecstride_y] =
                      alpha * sum[i] +
                      beta * d_y[grid_row_id * idxstride_y + i * vecstride_y];
                }
            }
            else
            {
                for (index_type i = 0; i < num_vectors; i++)
                {
                  d_y[grid_row_id * idxstride_y + i * vecstride_y] = alpha * sum[i];
                }
            }
          }
      }
    }

    /*--------------------------------------------------------------------------
    * hypreGPUKernel_CSRMatvecShuffle
    *
    * Templated SpMV device kernel based of warp-shuffle reduction.
    * Uses groups of K threads per row
    *
    * Template parameters:
    *   1) K:  number of threads working on a single row. K = 2, 4, 8, 16, 32
    *   2) F:  fill-mode. See hypreDevice_CSRMatrixMatvec for supported values
    *   3) NV: number of vectors (> 1 for multi-component vectors)
    *   4) T:  data type of matrix/vector coefficients
    *--------------------------------------------------------------------------*/

    template <index_type F, index_type K, index_type NV, typename T>
    __global__ void
    //__launch_bounds__(512, 1)
    hypreGPUKernel_CSRMatvecShuffle(hypre_DeviceItem &item,
                                    index_type         num_rows,
                                    index_type         num_vectors,
                                    index_type        *row_id,
                                    index_type         idxstride_x,
                                    index_type         idxstride_y,
                                    index_type         vecstride_x,
                                    index_type         vecstride_y,
                                    T                 alpha,
                                    index_type        *d_ia,
                                    index_type        *d_ja,
                                    T                *d_a,
                                    T                *d_x,
                                    T                 beta,
                                    T                *d_y )
    {
    #if defined(HYPRE_USING_SYCL)
      index_type grid_ngroups  = item.get_group_range(2) * (HYPRE_SPMV_BLOCKDIM / K);
      index_type grid_group_id = __shfl_sync(et_group(2) * HYPRE_SPMV_BLOCKDIM + item.get_local_id(2)) / K;
      index_type group_lane    = item.get_local_id(2) & (K - 1);
    #else
      const index_type  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
      index_type        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
      const index_type  group_lane    = threadIdx.x & (K - 1);
    #endif

      for (; __any_sync(0xffffffff, grid_group_id < num_rows);
            grid_group_id += grid_ngroups)
      {
        {
          index_type grid_row_id = -1, p = 0, q = 0;

          if (row_id)
          {
            if (grid_group_id < num_rows && group_lane == 0)
            {
                grid_row_id = __ldg(&row_id[grid_group_id]);
            }
            grid_row_id = __shfl_sync(0xffffffff, grid_row_id, 0, K);
          }
          else
          {
            grid_row_id = grid_group_id;
          }

          if (grid_group_id < num_rows && group_lane < 2)
          {
            p = __ldg(&d_ia[grid_row_id + group_lane]);
          }
          q = __shfl_sync(0xffffffff, p, 1, K);
          p = __shfl_sync(0xffffffff, p, 0, K);

          T sum[NV] = {T(0)};
    #if HYPRE_SPMV_VERSION == 1
    #pragma unroll 1
          for (p += group_lane; p < q; p += K * 2)
          {
            HYPRE_SPMV_ADD_SUM(p, NV)
            if (p + K < q)
            {
                HYPRE_SPMV_ADD_SUM((p + K), NV)
            }
          }
    #elif HYPRE_SPMV_VERSION == 2
    #pragma unroll 1
          for (p += group_lane; __any_sync(0xffffffff, p < q); p += K)
          {
            if (p < q)
            {
                HYPRE_SPMV_ADD_SUM(p, NV)
            }
          }
    #else
    #pragma unroll 1
          for (p += group_lane;  p < q; p += K)
          {
            HYPRE_SPMV_ADD_SUM(p, NV)
          }
    #endif

          // parallel reduction
          for (index_type i = 0; i < NV; i++)
          {
            for (index_type d = K / 2; d > 0; d >>= 1)
            {
                sum[i] += __shfl_down_sync(0xffffffff, sum[i], d);
            }
          }

          if (grid_group_id < num_rows && group_lane == 0)
          {
            for (index_type i = 0; i < NV; i++)
            {
              d_y[grid_row_id * idxstride_y + i * vecstride_y] = sum[i];
            }
            // if (beta)
            // {
            //     for (index_type i = 0; i < NV; i++)
            //     {
            //       d_y[grid_row_id * idxstride_y + i * vecstride_y] =
            //           alpha * sum[i] +
            //           beta * d_y[grid_row_id * idxstride_y + i * vecstride_y];
            //     }
            // }
            // else
            // {
            //     for (index_type i = 0; i < NV; i++)
            //     {
            //       d_y[grid_row_id * idxstride_y + i * vecstride_y] = alpha * sum[i];
            //     }
            // }
          }
        }
      }
    }

    int RandomizedConjugateGradientCuda::hypreDevice_CSRMatrixMatvec(matrix::Csr* A, vector::Vector* X, vector::Vector* result)
    {
      hypre_DeviceItem item = nullptr;
      index_type  num_vectors = X->getNumVectors();
      index_type  num_rows = A->getNumRows();
      index_type *rowid = nullptr;
      index_type  num_nonzeros = A->getNnz();
      index_type  idxstride_x = 1;
      index_type  idxstride_y = 1;
      index_type  vecstride_x = num_rows;
      index_type  vecstride_y = num_rows;
      real_type          alpha = 1.0;
      index_type *d_ia = A->getRowData(memory::DEVICE);
      index_type *d_ja = A->getColData(memory::DEVICE);
      real_type         *d_a = A->getValues(memory::DEVICE);
      real_type         *d_x = X->getData(memory::DEVICE);
      real_type          beta 
      = 0.0;
      real_type         *d_y = result->getData(memory::DEVICE);
      assert(d_ia && d_ja && d_x && d_y);

      constexpr int F = HYPRE_SPMV_FILL_WHOLE;

      const index_type avg_rownnz = (num_nonzeros + num_rows - 1) / num_rows;

      static constexpr index_type group_sizes[5] = {32, 16, 8, 4, 4};

      static constexpr index_type unroll_depth[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

      static index_type avg_rownnz_lower_bounds[5] = {64, 32, 16, 8, 0};

      static index_type num_groups_per_block[5] = { HYPRE_SPMV_BLOCKDIM / group_sizes[0],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[1],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[2],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[3],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[4]
                                                  };

      const dim3 bDim = dim3(HYPRE_SPMV_BLOCKDIM);

      /* Select execution path */
      switch (num_vectors)
      {
          case unroll_depth[1]:
            HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[1]);
            break;

          case unroll_depth[2]:
            HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[2]);
            break;

          // case unroll_depth[3]:
          //   HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[3]);
          //   break;

          case unroll_depth[4]:
            HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[4]);
            break;

          // case unroll_depth[5]:
          //   HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[5]);
          //   break;

          // case unroll_depth[6]:
          //   HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[6]);
          //   break;

          // case unroll_depth[7]:
          //   HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[7]);
          //   break;

          case unroll_depth[8]:
            HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffle, unroll_depth[8]);
            break;

          default:
            HYPRE_SPMV_GPU_LAUNCH(hypreGPUKernel_CSRMatvecShuffleGT8, unroll_depth[8]);
            break;
      }

      cudaDeviceSynchronize();
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess) {
          printf("Kernel error: %s\n", cudaGetErrorString(error));
      }

      return error;
    }
  } // namespace hykkt
} // namespace ReSolve
