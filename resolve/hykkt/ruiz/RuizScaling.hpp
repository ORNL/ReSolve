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
      RuizScaling(index_type          n,
                  index_type          total_n,
                  memory::MemorySpace memspace);
      ~RuizScaling();

      void            addMatrixData(matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr);
      void            addRhsData(vector::Vector* rhs_top, vector::Vector* rhs_bottom);
      vector::Vector* getAggregateScalingVector() const;
      void            scale(index_type num_iterations);

    private:
      index_type n_;       // Size of hessian (block [1,1])
      index_type total_n_; // Total matrix size

      // Matrix data
      matrix::Csr*    hes_;        // Hessian matrix
      matrix::Csr*    jac_;        // Jacobian matrix
      matrix::Csr*    jac_tr_;     // Transposed Jacobian matrix
      vector::Vector* rhs_top_;    // Top of the right-hand side vector
      vector::Vector* rhs_bottom_; // Bottom of the right-hand side vector

      vector::Vector* scaling_vector_;           // Scaling vector data for the current iteration
      vector::Vector* aggregate_scaling_vector_; // Cumulative scaling vector

      RuizScalingKernelImpl* kernelImpl_;

      memory::MemorySpace memspace_;

      void resetScaling();
      void allocateWorkspace();
      void deallocateWorkspace();
    };
  } // namespace hykkt
} // namespace ReSolve
