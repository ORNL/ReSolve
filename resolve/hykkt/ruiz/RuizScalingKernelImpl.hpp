#pragma once

#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace hykkt
  {

    class RuizScalingKernelImpl
    {
    public:
      using index_type = ReSolve::index_type;
      using real_type  = ReSolve::real_type;

      virtual void adaptRowMax(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* hes_tr_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* scaling_vector) = 0;

      virtual void adaptDiagScale(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* hes_tr_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, real_type* scaling_vector) = 0;
    };

  } // namespace hykkt
} // namespace ReSolve
