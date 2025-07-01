#pragma once
#include "RuizScalingKernelImpl.hpp"

namespace ReSolve
{
  namespace hykkt
  {

    class RuizScalingKernelsCUDA : public RuizScalingKernelImpl
    {
    public:
      void adaptRowMax(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, const real_type* hes_v, const index_type* jac_i, const index_type* jac_j, const real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, const real_type* jac_tr_v, real_type* scaling_vector);

      void adaptDiagScale(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, real_type* hes_v, const index_type* jac_i, const index_type* jac_j, real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, const real_type* scaling_vector);
    };

  } // namespace hykkt
} // namespace ReSolve
