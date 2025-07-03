#pragma once
#include "RuizScalingKernelImpl.hpp"

namespace ReSolve
{
  namespace hykkt
  {

    class RuizScalingKernelsCPU : public RuizScalingKernelImpl
    {
    public:
      void adaptRowMax(index_type n_hes, index_type n_total, matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr, vector::Vector* scaling_vector);

      void adaptDiagScale(index_type n_hes, index_type n_total, matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr, vector::Vector* rhs_top, vector::Vector* rhs_bottom, vector::Vector* aggregate_scaling_vector, vector::Vector* scaling_vector);
    };

  } // namespace hykkt
} // namespace ReSolve
