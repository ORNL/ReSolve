#include "MultiBasisParallelConjugateGradientHip.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <chrono>
#include <type_traits>

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    constexpr index_type WAVEFRONT_SIZE = 64;
    
    namespace kernels
    {
      template <index_type k>
      __device__ __forceinline__ index_type indexLowerTriangular(index_type i, index_type j)
      {
        return (j * (2 * k - 1 - j)) / 2 + i;
      }

      // ANDREW TODO: use this everywhere
      __device__ __forceinline__ real_type warpReduceSum(real_type val)
      {
        #pragma unroll
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        {
          val += __shfl_down(val, offset);
        }
        return val;
      }

      __device__ inline void gridReduceSum(const cooperative_groups::grid_group& grid,
                                           const real_type* __restrict__ local_vals,
                                           real_type* __restrict__ result,
                                           index_type length,
                                           real_type* __restrict__ workspace_global,
                                           real_type* __restrict__ workspace_shared)
      {
        if (grid.thread_rank() < length)
        {
          workspace_global[grid.thread_rank()] = 0.0;
        }
        grid.sync();

        index_type lane = threadIdx.x % WAVEFRONT_SIZE;
        index_type wavefront = threadIdx.x / WAVEFRONT_SIZE;
        index_type num_wavefronts = blockDim.x / WAVEFRONT_SIZE;

        // Reduce in wavefront
        for (int i = 0; i < length; ++i)
        {
          real_type wavefront_sum = warpReduceSum(local_vals[i]);
          if (lane == 0)
          {
            workspace_shared[wavefront * length + i] = wavefront_sum;
          }
        }
        __syncthreads();

        if (threadIdx.x < length)
        {
          // Reduce in block
          real_type block_sum = 0.0;
          for (int i = 0; i < num_wavefronts; i++)
          {
            block_sum += workspace_shared[i * length + threadIdx.x];
          }
          // Reduce globally
          atomicAdd(&workspace_global[threadIdx.x], block_sum);
        }

        grid.sync();

        // Broadcast
        for (int i = 0; i < length; i++)
        {
          result[i] = workspace_global[i];
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
        warpReduceSum(r_sq);
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
          index_type target_index = __shfl_down(best_basis_local, offset, k);
          real_type target_norm = __shfl_down(sq_norm_local, offset, k);
          if (target_norm < sq_norm_local)
          {
            best_basis_local = target_index;
            sq_norm_local = target_norm;
          }
        }

        if (thread == 0)
        {
          *best_basis = best_basis_local;
          sq_norms[0] = sqrt(sq_norm_local);
        }
      }

      template <index_type k>
      __global__ void qr(real_type* __restrict__ Q, real_type* __restrict__ R, index_type n)
      {
        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        index_type thread = blockIdx.x * blockDim.x + threadIdx.x;
        index_type stride = gridDim.x * blockDim.x;

        __shared__ real_type R_shared[k * k];
        __shared__ real_type tau_shared[k];
        __shared__ real_type reduction_workspace[4 * k];

        for (index_type idx = threadIdx.x; idx < k * k; idx += blockDim.x)
        {
          R_shared[idx] = 0.0;
        }
        __syncthreads();

        // Forward pass
        for (index_type i = 0; i < k; i++)
        {
          // Square norm of column i from rows i to n-1
          real_type sq_local = 0.0;
          for (index_type row = thread; row < n; row += stride)
          {
            if (row >= i)
            {
              real_type val = Q[i * n + row];
              sq_local += val * val;
            }
          }

          real_type norm_sq;
          gridReduceSum(grid, &sq_local, &norm_sq, 1, R, reduction_workspace);

          if (grid.thread_rank() == 0)
          {
            real_type q_ii = Q[i * n + i];
            real_type norm_x = sqrt(norm_sq);

            real_type alpha = 0.0;
            real_type beta = 0.0;
            real_type tau = 0.0;

            if (norm_x > 0.0)
            {
              alpha = (q_ii >= 0.0) ? -norm_x : norm_x;
              beta = q_ii - alpha; // No cancellation error since q_ii and alpha have opposite signs
              tau = -beta / alpha;
            }

            // k >= 2, so we can reuse R for temporary storage. k=1 skips this kernel
            R[0] = q_ii;
            R[1] = alpha;
            R[2] = beta;
            R[3] = tau;
          }
          grid.sync();

          real_type alpha = R[1];
          real_type beta  = R[2];
          real_type tau   = R[3];

          if (threadIdx.x == 0)
          {
            tau_shared[i] = tau;
            R_shared[i * k + i] = alpha;
          }
          __syncthreads();

          // Normalize below the diagonal
          for (index_type row = thread; row < n; row += stride)
          {
            if (row > i)
            {
              if (beta == 0.0)
              {
                Q[i * n + row] = 0.0;
              }
              else
              {
                Q[i * n + row] /= beta;
              }
            }
          }
          grid.sync();

          // Update values above the diagonal
          int cols_remaining = k - (i + 1);
          if (cols_remaining > 0)
          {
            real_type p_local[16] = { 0.0 };

            for (index_type row = thread; row < n; row += stride)
            {
              if (row == i)
              {
                #pragma unroll // ANDREW TODO: test no unroll
                for (int j = i + 1; j < k; j++)
                {
                  p_local[j - (i + 1)] += 1.0 * Q[j * n + row];
                }
              }
              else if (row > i)
              {
                real_type v_val = Q[i * n + row];
                #pragma unroll
                for (int j = i + 1; j < k; j++)
                {
                  p_local[j - (i + 1)] += v_val * Q[j * n + row];
                }
              }
            }

            real_type p[16];
            gridReduceSum(grid, p_local, p, cols_remaining, R, reduction_workspace);

            for (index_type row = thread; row < n; row += stride)
            {
              if (row == i)
              {
                #pragma unroll
                for (int j = i + 1; j < k; ++j)
                {
                  real_type r = Q[j * n + row] - tau * 1.0 * p[j - (i + 1)]; // R[i, j]
                  Q[j * n + row] = r;
                  if (threadIdx.x == 0)
                  {
                    R_shared[j * k + i] = r;
                  }
                }
              }
              else if (row > i)
              {
                real_type v_val = Q[i * n + row];
                #pragma unroll
                for (int j = i + 1; j < k; ++j)
                {
                  Q[j * n + row] -= tau * v_val * p[j - (i + 1)];
                }
              }
            }
            grid.sync();
          }
        }

        // Backward pass
        for (int i = k - 1; i >= 0; i--)
        {
          real_type tau = tau_shared[i];
          int cols_remaining = k - (i + 1);

          real_type d[16] = { 0.0 };
          if (cols_remaining > 0)
          {
            real_type local_d[16] = { 0.0 };
            for (index_type row = thread; row < n; row += stride)
            {
              if (row == i)
              {
                #pragma unroll
                for (int c = i + 1; c < k; ++c)
                {
                  local_d[c - (i + 1)] += 1.0 * Q[c * n + row];
                }
              }
              else if (row > i)
              {
                real_type v_val = Q[i * n + row];
                #pragma unroll
                for (int c = i + 1; c < k; ++c)
                {
                  local_d[c - (i + 1)] += v_val * Q[c * n + row];
                }
              }
            }
            gridReduceSum(grid, local_d, d, cols_remaining, R, reduction_workspace);
          }

          for (index_type row = thread; row < n; row += stride)
          {
            real_type v_r = (row == i) ? 1.0 : ((row > i) ? Q[i * n + row] : 0.0);

            // col > i
            #pragma unroll
            for (int col = i + 1; col < k; ++col)
            {
              if (row >= i)
              {
                Q[col * n + row] -= tau * v_r * d[col - (i + 1)];
              }
            }

            // Column i
            if (row < i)
            {
              Q[i * n + row] = 0.0;
            }
            else if (row == i)
            {
              Q[i * n + row] = 1.0 - tau;
            }
            else
            {
              Q[i * n + row] = -tau * v_r;
            }
          }
          grid.sync();
        }

        // Writeback
        for (index_type idx = grid.thread_rank(); idx < k * k; idx += gridDim.x * blockDim.x)
        {
          index_type r = idx % k;
          index_type c = idx / k;
          if (r > c)
          {
            R[c * k + r] = R_shared[c * k + r];
          }
          else
          {
            R[c * k + r] = 0.0;
          }
        }
      }

      template <index_type k>
      __global__ void updateXR(real_type* __restrict__ Xi_inv,
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
        index_type i = threadIdx.x % k;
        index_type j = threadIdx.x / k;
        if ((i < k) && (j < k) && (i >= j))
        {
          A_shared[indexLowerTriangular<k>(i, j)] = A[j * k + i];
        }
        __syncwarp();

        // Do all of this inside one warp. One thread per entry
        // #pragma unroll 1
        for (index_type h = 0; h < k; h++)
        {
          if (threadIdx.x == 0)
          {
            A_shared[indexLowerTriangular<k>(h, h)] = sqrt(fmax(A_shared[indexLowerTriangular<k>(h, h)], 1e-16)); // Prevent sqrt of negative number or division by a tiny number
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

      // X = A^-1 * B => A * X = B => L * L^T * X = B, choleskySolve // ANDREW TODO: cleanup notation
      // A_shared now contains L
        __shared__ real_type X_shared[k * k];
        if (threadIdx.x < k * k)
        {
          X_shared[threadIdx.x] = B[threadIdx.x];
        }
        __syncwarp();

        //  L * Y = B
        if (threadIdx.x < k)
        {
          index_type col = threadIdx.x; // Rows and columns of B

          // #pragma unroll 1
          for (index_type row = 0; row < k; row++)
          {
            real_type y_local = X_shared[col * k + row];
            // #pragma unroll 1
            for (index_type i = 0; i < row; i++)
            {
              y_local -= X_shared[col * k + i] * A_shared[indexLowerTriangular<k>(row, i)]; // A_shared can be overridden. Here it contains Y
            }
            y_local /= A_shared[indexLowerTriangular<k>(row, row)];
            X_shared[col * k + row] = y_local;
          }
        }

        //  L^T * X = Y
        if (threadIdx.x < k)
        {
          index_type col = threadIdx.x; // Talking about rows and columns of B

          // #pragma unroll 1
          for (index_type row = k - 1; row >= 0; row--)
          {
            real_type y_local = X_shared[col * k + row];
            // #pragma unroll 1
            for (index_type i = row + 1; i < k; i++)
            {
              y_local -= X_shared[col * k + i] * A_shared[indexLowerTriangular<k>(i, row)]; // A_shared can be overridden. Here it contains Y
            }
            y_local /= A_shared[indexLowerTriangular<k>(row, row)];
            X_shared[col * k + row] = y_local;
          }
        }
        __syncwarp();
        
        if (threadIdx.x < k * k)
        {
          X[threadIdx.x] = X_shared[threadIdx.x];
        }
      }

      template <index_type k>
      __global__ void choleskySolve(real_type* X, const real_type* __restrict__ L, const real_type* B)
      {
      // X = A * B => A * X = B => L * L^T * X = B, choleskySolve // ANDREW TODO: cleanup notation

        __shared__ real_type L_shared[k * (k + 1) / 2]; // Only need this much for kxk symmetric matrix. fits in one wavefront
        {
          index_type i = threadIdx.x % k;
          index_type j = threadIdx.x / k;
          if ((i < k) && (j < k) && (i >= j))
          {
            L_shared[indexLowerTriangular<k>(i, j)] = L[j * k + i];
          }
        }

        __shared__ real_type X_shared[k * k];
        if (threadIdx.x < k * k)
        {
          X_shared[threadIdx.x] = B[threadIdx.x];
        }
        __syncwarp();

        //  L * Y = B
        if (threadIdx.x < k)
        {
          index_type col = threadIdx.x; // Rows and columns of B

          // #pragma unroll 1
          for (index_type row = 0; row < k; row++)
          {
            real_type y_local = X_shared[col * k + row];
            // #pragma unroll 1
            for (index_type i = 0; i < row; i++)
            {
              y_local -= X_shared[col * k + i] * L_shared[indexLowerTriangular<k>(row, i)]; // L_shared can be overridden. Here it contains Y
            }
            y_local /= L_shared[indexLowerTriangular<k>(row, row)];
            X_shared[col * k + row] = y_local;
          }
        }

        //  L^T * X = Y
        if (threadIdx.x < k)
        {
          index_type col = threadIdx.x; // Talking about rows and columns of B

          // #pragma unroll 1
          for (index_type row = k - 1; row >= 0; row--)
          {
            real_type y_local = X_shared[col * k + row];
            // #pragma unroll 1
            for (index_type i = row + 1; i < k; i++)
            {
              y_local -= X_shared[col * k + i] * L_shared[indexLowerTriangular<k>(i, row)]; // L_shared can be overridden. Here it contains Y
            }
            y_local /= L_shared[indexLowerTriangular<k>(row, row)];
            X_shared[col * k + row] = y_local;
          }
        }
        __syncwarp(); // check if syncwarp instead of syncthreads breaks anything
        
        if (threadIdx.x < k * k)
        {
          X[threadIdx.x] = X_shared[threadIdx.x];
        }
      }

      template <index_type k, index_type BLOCK_DIM>
      __global__ void updateP(real_type* __restrict__ P, const real_type* __restrict__ R, const real_type* __restrict__ Xi_Delta, index_type n)
      {
        index_type thread = blockIdx.x * BLOCK_DIM + threadIdx.x;

        __shared__ real_type Xi_Delta_shared[k * k];
        if (threadIdx.x < k * k)
        {
          Xi_Delta_shared[threadIdx.x] = Xi_Delta[threadIdx.x];
        }
      
        __shared__ real_type P_shared[BLOCK_DIM * k];
        __shared__ real_type result_shared[BLOCK_DIM * k];
        // Compute P = R + P * Xi_Delta
        if (blockIdx.x < (n + BLOCK_DIM - 1) / BLOCK_DIM)
        {
          // Cache a horizontally-sliced block of P for coalesced reads
          for (index_type i = threadIdx.x; i < BLOCK_DIM * k; i += BLOCK_DIM)
          {
            index_type block_row = i % BLOCK_DIM;
            index_type global_row = block_row + blockIdx.x * BLOCK_DIM;
            index_type col = i / BLOCK_DIM;
            if (global_row < n)
            {
              P_shared[i] = P[col * n + global_row];
            } else {
              P_shared[i] = 0.0;
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
                dot += P_shared[i * BLOCK_DIM + threadIdx.x] * Xi_Delta_shared[i + j * k];
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
              P[col * n + global_row] = R[col * n + global_row] - result_shared[i];
            }
            result_shared[i] = 0.0;
          }
        }
      }

      /**
       * The point of this is to turn on/off loop unrolling based on the template argument k. C++11 doesn't support partial template specialization,
       * so this is the workaround. k=8 needs loop unrolling off for HIP due to register pressure.
       */
      template <index_type k, bool Unroll>
      __device__ __forceinline__ typename std::enable_if<!Unroll>::type multTSMTTSMLoop(const real_type* A, const real_type* B, real_type* __restrict__ C_shared, index_type row, index_type n)
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

      template <index_type k, bool Unroll>
      __device__ __forceinline__ typename std::enable_if<Unroll>::type multTSMTTSMLoop(const real_type* A, const real_type* B, real_type* __restrict__ C_shared, index_type row, index_type n)
      {
        real_type a_row[k];
        real_type b_row[k];
        
        #pragma unroll 1
        for (index_type col = 0; col < k; col++)
        {
          a_row[col] = A[col * n + row];
          b_row[col] = B[col * n + row];
        }

        #pragma unroll 1
        for (index_type i = threadIdx.x; i < threadIdx.x + k * k; i++)
        {
          index_type idx = i % (k * k);
          atomicAdd(&C_shared[idx], a_row[idx % k] * b_row[idx / k]);
        }
      }

      template <index_type k> // ANDREW TODO: rename template "k" to "K" (capitalized)
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
          constexpr bool unroll = (k <= 4);
          multTSMTTSMLoop<k, unroll>(A, B, C_shared, row, n);
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
        // Compute S = W + P * Zeta^T
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
    
    MultiBasisParallelConjugateGradientHip::MultiBasisParallelConjugateGradientHip(VectorHandler* vector_handler)
    {
      vector_handler_ = vector_handler;

      int device_id = 0;
      hipDeviceProp_t properties;
      hipGetDeviceProperties(&properties, device_id);
      num_sms_     = properties.multiProcessorCount; // 110
      num_threads_ = num_sms_ * properties.maxThreadsPerMultiProcessor; // 225280
      printf("Total SMs: %d, total threads: %d\n", num_sms_, num_threads_);
    }

    MultiBasisParallelConjugateGradientHip::~MultiBasisParallelConjugateGradientHip()
    {
      mem_.deleteOnDevice(d_best_basis_);
      mem_.deleteOnDevice(d_sq_norms_);
    }

    int MultiBasisParallelConjugateGradientHip::setup(index_type k)
    {
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

    int MultiBasisParallelConjugateGradientHip::SpMM(matrix::Csr* A, vector::Vector* X, vector::Vector* result)
    {
      return spmm_hypre_.hypreDevice_CSRMatrixMatvec(A, X, result);
    }
    
    int MultiBasisParallelConjugateGradientHip::bestBasis(vector::Vector* R, index_type* h_best_basis, real_type* h_best_basis_norm)
    {
      index_type n = R->getSize();
      index_type k = R->getNumVectors();

      dim3       block_size(WAVEFRONT_SIZE, k);
      int        num_blocks = num_sms_ * 4; // Pretty arbitrary. Tune this later
      hipMemsetAsync(d_sq_norms_, 0.0, k * sizeof(real_type));
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
      default:
        return 1;
      }

      hipMemcpyAsync(h_best_basis, d_best_basis_, sizeof(index_type), hipMemcpyDeviceToHost);
      hipMemcpyAsync(h_best_basis_norm, d_sq_norms_, sizeof(real_type), hipMemcpyDeviceToHost);
      hipDeviceSynchronize();

      return 0;
    }

  int MultiBasisParallelConjugateGradientHip::qr(vector::Vector* Q, vector::Vector* R, memory::MemorySpace memspace)
  {
    index_type n = Q->getSize();
    index_type k = Q->getNumVectors();

    if (k == 1)
    {
        real_type Q_norm = vector_handler_->norm(Q, memory::DEVICE);
        vector_handler_->scal(1 / Q_norm, Q, memory::DEVICE);
        R->setToConst(Q_norm, memory::DEVICE);
        return 0;
    }

    real_type* d_Q = Q->getData(memory::DEVICE);
    real_type* d_R = R->getData(memory::DEVICE);

    void* args[] = {
        (void*)&d_Q,
        (void*)&d_R,
        (void*)&n
    };

    void* kernel = nullptr;
    switch (k)
    {
    case 1: 
        kernel = (void*)kernels::qr<1>; 
        break;
    case 2: 
        kernel = (void*)kernels::qr<2>; 
        break;
    case 4: 
        kernel = (void*)kernels::qr<4>; 
        break;
    case 8: 
        kernel = (void*)kernels::qr<8>; 
        break;
    case 16:
        kernel = (void*)kernels::qr<16>;
        break;
    default:
        return 1;
    }

    int block_size = 256;
    int num_blocks = num_sms_;

    hipError_t status = hipLaunchCooperativeKernel(kernel, num_blocks, block_size, args, 0, 0);

    return status;
}
    
    // ANDREW TODO: separate out all of the cholesky functions. no reason to be combined. also, rename S to P
    int MultiBasisParallelConjugateGradientHip::updateXR(vector::Vector* Xi_inv, vector::Vector* Sigma, vector::Vector* S, vector::Vector* A_S, vector::Vector* Xi_Sigma, vector::Vector* X_res, vector::Vector* R_prec)
    {
      index_type n = A_S->getSize();
      index_type k = A_S->getNumVectors();

      int       block_size = WAVEFRONT_SIZE; // Must be at least k^2
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
      default:
        return 1;
      }

      block_size = 256; // Must be at least k^2
      num_blocks = num_sms_ * 64; // quite arbitrary

      switch (k)
      {
      case 1:
        kernels::updateXR<1><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      case 2:
        kernels::updateXR<2><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      case 4:
        kernels::updateXR<4><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
                                                         Sigma->getData(memory::DEVICE),
                                                         S->getData(memory::DEVICE),
                                                         A_S->getData(memory::DEVICE),
                                                         Xi_Sigma->getData(memory::DEVICE),
                                                         X_res->getData(memory::DEVICE),
                                                         R_prec->getData(memory::DEVICE),
                                                         n);
        break;
      case 8:
        kernels::updateXR<8><<<num_blocks, block_size>>>(Xi_inv->getData(memory::DEVICE),
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

    int MultiBasisParallelConjugateGradientHip::choleskyFactorizeSolve(vector::Vector* A, vector::Vector* B, vector::Vector* X)
    {
      index_type k = A->getSize();

      int       block_size = WAVEFRONT_SIZE; // Must be at least k^2
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
      default:
        return 1;
      }

      return 0;
    }

    // ... ANDREW TODO: rename R to Z
    int MultiBasisParallelConjugateGradientHip::updateP(vector::Vector* P, vector::Vector* R, vector::Vector* Xi_inv_chol, vector::Vector* Delta, memory::MemorySpace memspace)
    {
      index_type n = P->getSize();
      index_type k = P->getNumVectors();

      int       block_size_choleskySolve = WAVEFRONT_SIZE; // Must be at least k^2
      int       num_blocks = 1;
      auto start = std::chrono::steady_clock::now();

      switch (k)
      {
      case 1:
        kernels::choleskySolve<1><<<num_blocks, block_size_choleskySolve>>>(Delta->getData(memory::DEVICE),
                                                        Xi_inv_chol->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE));
        break;
      case 2:
        kernels::choleskySolve<2><<<num_blocks, block_size_choleskySolve>>>(Delta->getData(memory::DEVICE),
                                                        Xi_inv_chol->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE));
        break;
      case 4:
        kernels::choleskySolve<4><<<num_blocks, block_size_choleskySolve>>>(Delta->getData(memory::DEVICE),
                                                        Xi_inv_chol->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE));
        break;
      case 8:
        kernels::choleskySolve<8><<<num_blocks, block_size_choleskySolve>>>(Delta->getData(memory::DEVICE),
                                                        Xi_inv_chol->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE));
        break;
      default:
        return 1;
      }

      // One thread per row of P
      constexpr int block_size_updateP = 128; // Must be at least k^2.. TODO: TEST 256
      num_blocks = (n + block_size_updateP - 1) / block_size_updateP;

      switch (k)
      {
      case 1:
        kernels::updateP<1, block_size_updateP><<<num_blocks, block_size_updateP>>>(P->getData(memory::DEVICE),
                                                        R->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE),
                                                        n);
        break;
      case 2:
        kernels::updateP<2, block_size_updateP><<<num_blocks, block_size_updateP>>>(P->getData(memory::DEVICE),
                                                        R->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE),
                                                        n);
        break;
      case 4:
        kernels::updateP<4, block_size_updateP><<<num_blocks, block_size_updateP>>>(P->getData(memory::DEVICE),
                                                        R->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE),
                                                        n);
        break;
      case 8:
        kernels::updateP<8, block_size_updateP><<<num_blocks, block_size_updateP>>>(P->getData(memory::DEVICE),
                                                        R->getData(memory::DEVICE),
                                                        Delta->getData(memory::DEVICE),
                                                        n);
        break;
      default:
        return 1;
      }

      return 0;
    }

    // Tall & skinny matrix times transposed tall & skinny matrix
    // C = A^T * B
    int MultiBasisParallelConjugateGradientHip::multTSMTTSM(vector::Vector* A, vector::Vector* B, vector::Vector* C, memory::MemorySpace memspace)
    {
      index_type n = A->getSize();
      index_type k = A->getNumVectors();

      int       block_size = 256;
      int       num_blocks = num_sms_ * WAVEFRONT_SIZE;
      
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
      default:
        return 1;
      }

      return 0;
    }

    int MultiBasisParallelConjugateGradientHip::updateSSigma(vector::Vector* W, vector::Vector* S, vector::Vector* Zeta, vector::Vector* Sigma, memory::MemorySpace memspace)
    {
      index_type n = W->getSize();
      index_type k = W->getNumVectors();  

      // One thread per row for S = W + S * Zeta^T
      // One block for Sigma = Zeta * Sigma
      constexpr int block_size = 128; // Must be at least k^2
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
      default:
        return 1;
      }

      return 0;
    }

    int MultiBasisParallelConjugateGradientHip::preconditionDense(vector::Vector* A, vector::Vector* d)
    {
      index_type n = A->getSize();

      constexpr int block_size = 256;
      int num_blocks = (n * n + block_size - 1) / block_size;
      kernels::preconditionDense<<<num_blocks, block_size>>>(A->getData(memory::DEVICE), d->getData(memory::DEVICE), n);

      return 0;
    }
  } // namespace hykkt
} // namespace ReSolve
