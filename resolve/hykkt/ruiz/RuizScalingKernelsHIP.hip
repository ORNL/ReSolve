#include <hip/hip_runtime.h>

#include "RuizScalingKernelsHIP.hpp"

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    namespace kernels
    {
      /**
       * @brief HIP kernel implementation of adaptRowMax.
       *
       * @param[in] n_hes - Number of rows in the Hessian matrix.
       * @param[in] n_total - Total number of rows in the 2x2 block system.
       * @param[in] hes_i - Row pointers for the Hessian matrix.
       * @param[in] hes_j - Column indices for the Hessian matrix.
       * @param[in] hes_v - Values for the Hessian matrix.
       * @param[in] jac_i - Row pointers for the Jacobian matrix.
       * @param[in] jac_j - Column indices for the Jacobian matrix.
       * @param[in] jac_v - Values for the Jacobian matrix.
       * @param[in] jac_tr_i - Row pointers for the transposed Jacobian matrix.
       * @param[in] jac_tr_j - Column indices for the transposed Jacobian matrix.
       * @param[in] jac_tr_v - Values for the transposed Jacobian matrix.
       * @param[out] scaling_vector - Scaling vector output
       */
      __global__ void adaptRowMax(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, const real_type* hes_v, const index_type* jac_i, const index_type* jac_j, const real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, const real_type* jac_tr_v, real_type* scaling_vector)
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
            scaling_vector[i] = 1.0 / std::sqrt(max_l);
          }
          else
          {
            scaling_vector[i] = 1.0 / std::sqrt(max_u);
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
          scaling_vector[i] = 1.0 / std::sqrt(max_l);
        }
      }

      /**
       * @brief HIP kernel implementation for adaptRowMax.
       *
       * @param[in] n_hes - Number of rows in the Hessian matrix.
       * @param[in] n_total - Total number of rows in the 2x2 block system.
       * @param[in] hes_i - Row pointers for the Hessian matrix.
       * @param[in] hes_j - Column indices for the Hessian matrix.
       * @param[in,out] hes_v - Values for the Hessian matrix.
       * @param[in] jac_i - Row pointers for the Jacobian matrix.
       * @param[in] jac_j - Column indices for the Jacobian matrix.
       * @param[in,out] jac_v - Values for the Jacobian matrix.
       * @param[in] jac_tr_i - Row pointers for the transposed Jacobian matrix.
       * @param[in] jac_tr_j - Column indices for the transposed Jacobian matrix.
       * @param[in,out] jac_tr_v - Values for the transposed Jacobian matrix.
       * @param[in,out] rhs1 - Right-hand side vector for the top block.
       * @param[in,out] rhs2 - Right-hand side vector for the bottom block.
       * @param[in,out] aggregate_scaling_vector - Cumulative scaling vector to be updated.
       * @param[in] scaling_vector - Scaling vector computed in the current iteration.
       */
      __global__ void adaptDiagScale(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, real_type* hes_v, const index_type* jac_i, const index_type* jac_j, real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, const real_type* scaling_vector)
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

    /**
     * @brief HIP kernel wrapper of adaptRowMax. See RuizScalingKernelImpl.
     *
     * @post scaling_vector contains the scaling factors for the current iteration. See RuizScalingKernelImpl.hpp for details.
     *
     * @param[in] n_hes - Number of rows in the Hessian matrix.
     * @param[in] n_total - Total number of rows in the 2x2 block system.
     * @param[in] hes_i - Row pointers for the Hessian matrix.
     * @param[in] hes_j - Column indices for the Hessian matrix.
     * @param[in] hes_v - Values for the Hessian matrix.
     * @param[in] jac_i - Row pointers for the Jacobian matrix.
     * @param[in] jac_j - Column indices for the Jacobian matrix.
     * @param[in] jac_v - Values for the Jacobian matrix.
     * @param[in] jac_tr_i - Row pointers for the transposed Jacobian matrix.
     * @param[in] jac_tr_j - Column indices for the transposed Jacobian matrix.
     * @param[in] jac_tr_v - Values for the transposed Jacobian matrix.
     * @param[out] scaling_vector - Scaling vector output.
     */
    void RuizScalingKernelsHIP::adaptRowMax(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, const real_type* hes_v, const index_type* jac_i, const index_type* jac_j, const real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, const real_type* jac_tr_v, real_type* scaling_vector)
    {
      int num_blocks;
      int block_size = 256;
      num_blocks     = (n_total + block_size - 1) / block_size;
      kernels::adaptRowMax<<<num_blocks, block_size>>>(n_hes, n_total, hes_i, hes_j, hes_v, jac_i, jac_j, jac_v, jac_tr_i, jac_tr_j, jac_tr_v, scaling_vector);
    }

    /**
     * @brief HIP kernel implementation for adaptRowMax. See RuizScalingKernelImpl.
     * 
     * @pre adaptRowMax has been called to compute the scaling_vector.
     * @post The system is scaled. See RuizScalingKernelImpl.hpp for details.
     * 
     * @param[in] n_hes - Number of rows in the Hessian matrix.
     * @param[in] n_total - Total number of rows in the 2x2 block system.
     * @param[in] hes_i - Row pointers for the Hessian matrix.
     * @param[in] hes_j - Column indices for the Hessian matrix.
     * @param[in,out] hes_v - Values for the Hessian matrix to be scaled.
     * @param[in] jac_i - Row pointers for the Jacobian matrix.
     * @param[in] jac_j - Column indices for the Jacobian matrix.
     * @param[in,out] jac_v - Values for the Jacobian matrix to be scaled.
     * @param[in] jac_tr_i - Row pointers for the transposed Jacobian matrix.
     * @param[in] jac_tr_j - Column indices for the transposed Jacobian matrix.
     * @param[in,out] jac_tr_v - Values for the transposed Jacobian matrix to be scaled.
     * @param[in,out] rhs1 - Right-hand side vector for the top block.
     * @param[in,out] rhs2 - Right-hand side vector for the bottom block.
     * @param[in,out] aggregate_scaling_vector - Cumulative scaling vector to be updated.
     * @param[in] scaling_vector - Scaling vector computed in the current iteration.
     */
    void RuizScalingKernelsHIP::adaptDiagScale(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, real_type* hes_v, const index_type* jac_i, const index_type* jac_j, real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, const real_type* scaling_vector)
    {
      int block_size = 256;
      int num_blocks = (n_total + block_size - 1) / block_size;
      kernels::adaptDiagScale<<<num_blocks, block_size>>>(n_hes, n_total, hes_i, hes_j, hes_v, jac_i, jac_j, jac_v, jac_tr_i, jac_tr_j, jac_tr_v, rhs1, rhs2, aggregate_scaling_vector, scaling_vector);
    }
  } // namespace hykkt
} // namespace ReSolve
