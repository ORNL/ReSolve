#include <cuda_runtime.h>

#include "RuizScalingKernelsCUDA.hpp"

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    namespace kernels
    {
      __global__ void adaptRowMax(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* jac_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* scaling_vector)
      {
        real_type  max_l = 0;
        real_type  max_u = 0;
        index_type i     = blockIdx.x * blockDim.x + threadIdx.x;
        index_type j;
        real_type  entry;
        if (i < n_hes)
        {
          for (j = hes_i[i]; j < hes_i[i + 1]; j++)
          {
            entry = fabs(hes_v[j]);
            if (entry > max_l)
            {
              max_l = entry;
            }
          }
          for (j = jac_tr_i[i]; j < jac_tr_i[i + 1]; j++)
          {
            entry = fabs(jac_tr_v[j]);
            if (entry > max_u)
            {
              max_u = entry;
            }
          }
          if (max_l > max_u)
          {
            scaling_vector[i] = 1.0 / sqrt(max_l);
          }
          else
          {
            scaling_vector[i] = 1.0 / sqrt(max_u);
          }
        }
        if (i >= n_hes && i < n_total)
        {
          for (j = jac_i[i - n_hes]; j < jac_i[i - n_hes + 1]; j++)
          {
            entry = fabs(jac_v[j]);
            if (entry > max_l)
            {
              max_l = entry;
            }
          }
          scaling_vector[i] = 1.0 / sqrt(max_l);
        }
      }

      __global__ void adaptDiagScale(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* jac_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, real_type* scaling_vector)
      {
        index_type i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_hes)
        {
          for (index_type j = hes_i[i]; j < hes_i[i + 1]; j++)
          {
            hes_v[j] *= scaling_vector[i] * scaling_vector[hes_j[j]];
          }
          for (index_type j = jac_tr_i[i]; j < jac_tr_i[i + 1]; j++)
          {
            jac_tr_v[j] *= scaling_vector[i] * scaling_vector[n_hes + jac_tr_j[j]];
          }
          rhs1[i] *= scaling_vector[i];
          aggregate_scaling_vector[i] *= scaling_vector[i];
        }
        if (i >= n_hes && i < n_total)
        {
          for (index_type j = jac_i[i - n_hes]; j < jac_i[i - n_hes + 1]; j++)
          {
            jac_v[j] *= scaling_vector[i] * scaling_vector[jac_j[j]];
          }
          rhs2[i - n_hes] *= scaling_vector[i];
          aggregate_scaling_vector[i] *= scaling_vector[i];
        }
      }
    } // namespace kernels

    void RuizScalingKernelsCUDA::adaptRowMax(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* jac_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* scaling_vector)
    {
      int num_blocks;
      int block_size = 256;
      num_blocks     = (n_total + block_size - 1) / block_size;
      kernels::adaptRowMax<<<num_blocks, block_size>>>(n_hes, n_total, hes_i, hes_j, hes_v, jac_i, jac_j, jac_v, jac_tr_i, jac_tr_j, jac_tr_v, scaling_vector);
    }

    void RuizScalingKernelsCUDA::adaptDiagScale(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* jac_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, real_type* scaling_vector)
    {
      int block_size = 256;
      int num_blocks = (n_total + block_size - 1) / block_size;
      kernels::adaptDiagScale<<<num_blocks, block_size>>>(n_hes, n_total, hes_i, hes_j, hes_v, jac_i, jac_j, jac_v, jac_tr_i, jac_tr_j, jac_tr_v, rhs1, rhs2, aggregate_scaling_vector, scaling_vector);
    }
  } // namespace hykkt
} // namespace ReSolve
