#include "RuizScalingKernelsCPU.hpp"

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    void hykkt::RuizScalingKernelsCPU::adaptRowMax(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* jac_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* scaling_vector)
    {
      for (index_type i = 0; i < n_hes; i++)
      {
        real_type max_l = 0;
        real_type max_u = 0;
        for (index_type j = hes_i[i]; j < hes_i[i + 1]; j++)
        {
          real_type entry = fabs(hes_v[j]);
          if (entry > max_l)
          {
            max_l = entry;
          }
        }
        for (index_type j = jac_tr_i[i]; j < jac_tr_i[i + 1]; j++)
        {
          real_type entry = fabs(jac_tr_v[j]);
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
      for (index_type i = n_hes; i < n_total; i++)
      {
        double max_l = 0;
        for (index_type j = jac_i[i - n_hes]; j < jac_i[i - n_hes + 1]; j++)
        {
          real_type entry = fabs(jac_v[j]);
          if (entry > max_l)
          {
            max_l = entry;
          }
        }
        scaling_vector[i] = 1.0 / sqrt(max_l);
      }
    }

    void hykkt::RuizScalingKernelsCPU::adaptDiagScale(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* jac_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, real_type* scaling_vector)
    {
      for (index_type i = 0; i < n_hes; i++)
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
      for (index_type i = n_hes; i < n_total; i++)
      {
        for (index_type j = jac_i[i - n_hes]; j < jac_i[i - n_hes + 1]; j++)
        {
          jac_v[j] *= scaling_vector[i] * scaling_vector[jac_j[j]];
        }
        rhs2[i - n_hes] *= scaling_vector[i];
        aggregate_scaling_vector[i] *= scaling_vector[i];
      }
    }
  } // namespace hykkt
} // namespace ReSolve
