#pragma once

#include "RuizScalingKernelImpl.hpp"
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    class RuizScaling
    {
    public:
      RuizScaling(index_type          num_iterations,
                 index_type          n,
                 index_type          total_n,
                 memory::MemorySpace memspace);
      ~RuizScaling();

      void       addHInfo(matrix::Csr* hes);
      void       addJInfo(matrix::Csr* jac);
      void       addJtInfo(matrix::Csr* jac_tr);
      void       addRhsTop(vector::Vector* rhs_top);
      void       addRhsBottom(vector::Vector* rhs_bottom);
      real_type* getAggregateScalingVector() const;
      void       scale();

    private:
      // Number of iterations
      index_type num_iterations_;
      // Size of hessian (block [1,1])
      index_type n_;
      // Total matrix size
      index_type total_n_;

      // Matrix data
      real_type*  hes_v_;
      index_type* hes_i_;
      index_type* hes_j_;
      real_type*  jac_v_;
      index_type* jac_i_;
      index_type* jac_j_;
      real_type*  jac_tr_v_;
      index_type* jac_tr_i_;
      index_type* jac_tr_j_;
      real_type*  rhs_top_;
      real_type*  rhs_bottom_;

      real_type* scaling_vector_;           // Scaling vector for the current iteration
      real_type* aggregate_scaling_vector_; // Cumulative scaling vector

      RuizScalingKernelImpl* kernelImpl_;

      memory::MemorySpace memspace_;
      MemoryHandler       mem_;

      void resetScaling();

      void allocateWorkspace();

      void deallocateWorkspace();
    };
  } // namespace hykkt
} // namespace ReSolve
