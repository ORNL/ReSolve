#pragma once

#include <cmath>

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

      virtual ~RuizScalingKernelImpl() = default;

      /**
       * @brief Compute one iteration of Ruiz scaling
       *
       * @post scaling_vector[i] = 1 / sqrt(max_val) where max_val is the maximum absolute
       *       value of the entries in the i-th row of the block matrix [hes jac^T; jac 0]
       *
       * @param[in] n_hes - Number of rows in the Hessian matrix.
       * @param[in] n_total - Total number of rows in the system.
       * @param[in] hes_i - Row pointers for the Hessian matrix.
       * @param[in] hes_j - Column indices for the Hessian matrix.
       * @param[in] hes_v - Values for the Hessian matrix.
       * @param[in] jac_i - Row pointers for the Jacobian matrix.
       * @param[in] jac_j - Column indices for the Jacobian matrix.
       * @param[in] jac_v - Values for the Jacobian matrix.
       * @param[in] jac_tr_i - Row pointers for the transposed Jacobian matrix.
       * @param[in] jac_tr_j - Column indices for the transposed Jacobian matrix.
       * @param[in] jac_tr_v - Values for the transposed Jacobian matrix.
       * @param[out] scaling_vector - Scaling vector to be updated.
       */
      virtual void adaptRowMax(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, const real_type* hes_v, const index_type* jac_i, const index_type* jac_j, const real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, const real_type* jac_tr_v, real_type* scaling_vector) = 0;

      /**
       * @brief Apply the scaling stored in scaling_vector to the matrices and rhs
       *
       * @post The left hand side is scaled from the left and right and the right hand side
       *       is scaled from the right by the scaling_vector.
       *
       * @param[in] n_hes - Number of rows in the Hessian matrix.
       * @param[in] n_total - Total number of rows in the system.
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
       * @param[in,out] aggregate_scaling_vector - Cumulative scaling vector.
       * @param[in] scaling_vector - Scaling vector for the current iteration.
       */
      virtual void adaptDiagScale(index_type n_hes, index_type n_total, const index_type* hes_i, const index_type* hes_j, real_type* hes_v, const index_type* jac_i, const index_type* jac_j, real_type* jac_v, const index_type* jac_tr_i, const index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, const real_type* scaling_vector) = 0;
    };

  } // namespace hykkt
} // namespace ReSolve
