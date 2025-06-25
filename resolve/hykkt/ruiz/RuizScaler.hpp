#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

#include "RuizScalingKernelImpl.hpp"

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    class RuizScaler
    {
    public:
      RuizScaler(index_type          num_iterations,
                 index_type          n,
                 index_type          total_n,
                 memory::MemorySpace memspace);
      ~RuizScaler();

      /**
       *  @brief Add Hessian information to the Ruiz scaling handler.
       *  @param hes[in] - Pointer to CSR matrix representing the Hessian.
       */
      void addHInfo(matrix::Csr* hes);

      /**
       *  @brief Add Jacobian information to the Ruiz scaling handler.
       *  @param jac[in] - Pointer to CSR matrix representing the Jacobian.
       */
      void addJInfo(matrix::Csr* jac);

      /**
       *  @brief Add Jacobian transpose information to the Ruiz scaling handler.
       *  @param jac_tr[in] - Pointer to CSR matrix representing the transpose of the Jacobian.
       */
      void addJtInfo(matrix::Csr* jac_tr);

      /**
       *  @brief Add right-hand side vector to the Ruiz scaling handler.
       *  @param rhs1[in] - Pointer to the top right-hand side vector.
       */
      void addRhsTop(vector::Vector* rhs_top);

      /**
       *  @brief Add right-hand side vector to the Ruiz scaling handler.
       *  @param rhs2[in] - Pointer to the bottom right-hand side vector.
       */
      void addRhsBottom(vector::Vector* rhs_bottom);

      /**
       *  @brief Get the scaling vector.
       *  @return Pointer to the scaling vector.
       */
      real_type* getAggregateScalingVector() const;

      /**
       *  @brief Compute the Ruiz scaling.
       */
      void scale();

    private:
      index_type num_iterations_;
      index_type n_;
      index_type total_n_;

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

      real_type* scaling_vector_;
      real_type* aggregate_scaling_vector_;

      RuizScalingKernelImpl* kernelImpl_;

      memory::MemorySpace memspace_;
      MemoryHandler       mem_;

      void resetScaling();

      void allocateWorkspace();

      void deallocateWorkspace();
    };
  } // namespace hykkt
} // namespace ReSolve
