//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/******************************************************************************
* Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
*
* SPDX-License-Identifier: (Apache-2.0 OR MIT)
******************************************************************************/

#include "SpMMHip.hpp"

// "constexpr if" requires C++17. Suppress warnings about this
#pragma GCC diagnostic ignored "-Wc++17-compat"
#pragma clang diagnostic ignored "-Wc++17-extensions"

namespace ReSolve
{
  namespace hykkt
  {
    using hypre_DeviceItem = void*;

    #define HYPRE_SPMV_BLOCKDIM 512
    #define HYPRE_SPMV_VERSION 1
    #define HYPRE_SPMV_FILL_STRICT_LOWER -2
    #define HYPRE_SPMV_FILL_LOWER -1
    #define HYPRE_SPMV_FILL_WHOLE 0
    #define HYPRE_SPMV_FILL_UPPER 1
    #define HYPRE_SPMV_FILL_STRICT_UPPER 2

    #define HYPRE_SPMV_ADD_SUM(p, nv)                                                                        \
      {                                                                                                  \
        const index_type col = __ldg(&d_ja[p]);                                                          \
        if constexpr (F == HYPRE_SPMV_FILL_WHOLE)                                                        \
        {                                                                                                \
            const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                                   \
            for (index_type i = 0; i < nv; i++)                                                          \
            {                                                                                            \
              sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);                          \
            }                                                                                            \
        }                                                                                                \
        else if constexpr (F == HYPRE_SPMV_FILL_LOWER)                                                   \
        {                                                                                                \
            if (col <= grid_group_id)                                                                    \
            {                                                                                            \
              const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                                 \
              for (index_type i = 0; i < nv; i++)                                                        \
              {                                                                                          \
                  sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);                      \
              }                                                                                          \
            }                                                                                            \
        }                                                                                                \
        else if constexpr (F == HYPRE_SPMV_FILL_UPPER)                                                   \
        {                                                                                                \
            if (col >= grid_group_id)                                                                    \
            {                                                                                            \
              const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                                 \
              for (index_type i = 0; i < nv; i++)                                                        \
              {                                                                                          \
                  sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);                      \
              }                                                                                          \
            }                                                                                            \
        }                                                                                                \
        else if constexpr (F == HYPRE_SPMV_FILL_STRICT_LOWER)                                            \
        {                                                                                                \
            if (col < grid_group_id)                                                                     \
            {                                                                                            \
              const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                                 \
              for (index_type i = 0; i < nv; i++)                                                        \
              {                                                                                          \
                  sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);                      \
              }                                                                                          \
            }                                                                                            \
        }                                                                                                \
        else if constexpr (F == HYPRE_SPMV_FILL_STRICT_UPPER)                                            \
        {                                                                                                \
            if (col > grid_group_id)                                                                     \
            {                                                                                            \
              const real_type val = d_a ? __ldg(&d_a[p]) : real_type(1);                                 \
              for (index_type i = 0; i < nv; i++)                                                        \
              {                                                                                          \
                  sum[i] += val * __ldg(&d_x[col * idxstride_x + i * vecstride_x]);                      \
              }                                                                                          \
            }                                                                                            \
        }                                                                                                \
      }

    namespace kernels
    {          
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
        const index_type  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
        index_type        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
        const index_type  group_lane    = threadIdx.x & (K - 1);
        T sum[64];

        for (; __any_sync(0xffffffffffffffff, grid_group_id < num_rows);
              grid_group_id += grid_ngroups)
        {
            index_type grid_row_id = -1, p = 0, q = 0;

            if (row_id)
            {
              if (grid_group_id < num_rows && group_lane == 0)
              {
                  grid_row_id = __ldg(&row_id[grid_group_id]);
              }
              grid_row_id = __shfl_sync(0xffffffffffffffff, grid_row_id, 0, K);
            }
            else
            {
              grid_row_id = grid_group_id;
            }

            if (grid_group_id < num_rows && group_lane < 2)
            {
              p = __ldg(&d_ia[grid_row_id + group_lane]);
            }
            q = __shfl_sync(0xffffffffffffffff, p, 1, K);
            p = __shfl_sync(0xffffffffffffffff, p, 0, K);

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
                  sum[i] += __shfl_down_sync(0xffffffffffffffff, sum[i], d);
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
        const index_type  grid_ngroups  = gridDim.x * (HYPRE_SPMV_BLOCKDIM / K);
        index_type        grid_group_id = (blockIdx.x * HYPRE_SPMV_BLOCKDIM + threadIdx.x) / K;
        const index_type  group_lane    = threadIdx.x & (K - 1);

        for (; __any_sync(0xffffffffffffffff, grid_group_id < num_rows);
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
              grid_row_id = __shfl_sync(0xffffffffffffffff, grid_row_id, 0, K);
            }
            else
            {
              grid_row_id = grid_group_id;
            }

            if (grid_group_id < num_rows && group_lane < 2)
            {
              p = __ldg(&d_ia[grid_row_id + group_lane]);
            }
            q = __shfl_sync(0xffffffffffffffff, p, 1, K);
            p = __shfl_sync(0xffffffffffffffff, p, 0, K);

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
            for (p += group_lane; __any_sync(0xffffffffffffffff, p < q); p += K)
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
                  sum[i] += __shfl_down_sync(0xffffffffffffffff, sum[i], d);
              }
            }

            if (grid_group_id < num_rows && group_lane == 0)
            {
              for (index_type i = 0; i < NV; i++)
              {
                d_y[grid_row_id * idxstride_y + i * vecstride_y] = sum[i];
              }
            }
          }
        }
      }
    } // namespace kernels

    
      #define HYPRE_SPMV_GPU_LAUNCH(kernel, nv)                                                          \
        if (avg_rownnz >= avg_rownnz_lower_bounds[0])                                                    \
        {                                                                                                \
            const dim3 gDim =                                                                            \
              dim3((num_rows + num_groups_per_block[0] - 1) / num_groups_per_block[0]);                  \
            HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[0], nv, real_type>),                                \
                              gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,     \
                              idxstride_y, vecstride_x, vecstride_y, alpha,                              \
                              d_ia, d_ja, d_a, d_x, beta, d_y );                                         \
        }                                                                                                \
        else if (avg_rownnz >= avg_rownnz_lower_bounds[1])                                               \
        {                                                                                                \
            const dim3 gDim =                                                                            \
              dim3((num_rows + num_groups_per_block[1] - 1) / num_groups_per_block[1]);                  \
            HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[1], nv, real_type>),                                \
                              gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,     \
                              idxstride_y, vecstride_x, vecstride_y, alpha,                              \
                              d_ia, d_ja, d_a, d_x, beta, d_y );                                         \
        }                                                                                                \
        else if (avg_rownnz >= avg_rownnz_lower_bounds[2])                                               \
        {                                                                                                \
            const dim3 gDim =                                                                            \
              dim3((num_rows + num_groups_per_block[2] - 1) / num_groups_per_block[2]);                  \
            HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[2], nv, real_type>),                                \
                              gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,     \
                              idxstride_y, vecstride_x, vecstride_y, alpha,                              \
                              d_ia, d_ja, d_a, d_x, beta, d_y );                                         \
        }                                                                                                \
        else if (avg_rownnz >= avg_rownnz_lower_bounds[3])                                               \
        {                                                                                                \
            const dim3 gDim =                                                                            \
              dim3((num_rows + num_groups_per_block[3] - 1) / num_groups_per_block[3]);                  \
            HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[3], nv, real_type>),                                \
                              gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,     \
                              idxstride_y, vecstride_x, vecstride_y, alpha,                              \
                              d_ia, d_ja, d_a, d_x, beta, d_y );                                         \
        }                                                                                                \
        else if (avg_rownnz >= avg_rownnz_lower_bounds[4])                                               \
        {                                                                                                \
            const dim3 gDim =                                                                            \
              dim3((num_rows + num_groups_per_block[4] - 1) / num_groups_per_block[4]);                  \
            HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[4], nv, real_type>),                                \
                              gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,     \
                              idxstride_y, vecstride_x, vecstride_y, alpha,                              \
                              d_ia, d_ja, d_a, d_x, beta, d_y );                                         \
        }                                                                                                \
        else                                                                                             \
        {                                                                                                \
            const dim3 gDim =                                                                            \
              dim3((num_rows + num_groups_per_block[5] - 1) / num_groups_per_block[5]);                  \
            HYPRE_GPU_LAUNCH( (kernel<F, group_sizes[5], nv, real_type>),                                \
                              gDim, bDim, item, num_rows, num_vectors, rowid = nullptr, idxstride_x,     \
                              idxstride_y, vecstride_x, vecstride_y, alpha,                              \
                              d_ia, d_ja, d_a, d_x, beta, d_y );                                         \
        }

    #define HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) kernel_name<<<gridsize, blocksize>>>(__VA_ARGS__)

    int SpMMHip::hypreDevice_CSRMatrixMatvec(matrix::Csr* A, vector::Vector* X, vector::Vector* result)
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
      real_type          beta = 0.0;
      real_type         *d_y = result->getData(memory::DEVICE);

      constexpr int F = HYPRE_SPMV_FILL_WHOLE;

      const index_type avg_rownnz = (num_nonzeros + num_rows - 1) / num_rows;

      static constexpr index_type group_sizes[6] = {64, 32, 16, 8, 4, 4};

      static constexpr index_type unroll_depth[4] = {1, 2, 4, 8};

      static index_type avg_rownnz_lower_bounds[6] = {128, 64, 32, 16, 8, 0};

      static index_type num_groups_per_block[6] = { HYPRE_SPMV_BLOCKDIM / group_sizes[0],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[1],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[2],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[3],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[4],
                                                    HYPRE_SPMV_BLOCKDIM / group_sizes[5]
                                                  };

      const dim3 bDim = dim3(HYPRE_SPMV_BLOCKDIM);

      /* Select execution path */
      switch (num_vectors)
      {
          case unroll_depth[0]:
            HYPRE_SPMV_GPU_LAUNCH(kernels::hypreGPUKernel_CSRMatvecShuffle, unroll_depth[0]);
            break;
          case unroll_depth[1]:
            HYPRE_SPMV_GPU_LAUNCH(kernels::hypreGPUKernel_CSRMatvecShuffle, unroll_depth[1]);
            break;
          case unroll_depth[2]:
            HYPRE_SPMV_GPU_LAUNCH(kernels::hypreGPUKernel_CSRMatvecShuffle, unroll_depth[2]);
            break;
          case unroll_depth[3]:
            HYPRE_SPMV_GPU_LAUNCH(kernels::hypreGPUKernel_CSRMatvecShuffle, unroll_depth[3]);
            break;

          default:
            HYPRE_SPMV_GPU_LAUNCH(kernels::hypreGPUKernel_CSRMatvecShuffleGT8, unroll_depth[3]);
            break;
      }

      return 0;
    }
  } // namespace hykkt
} // namespace ReSolve