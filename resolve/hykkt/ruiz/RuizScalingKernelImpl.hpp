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
       * @brief Determines the correct scaling corresponding to one iteration
       *        of Ruiz scaling on matrix with form [hes jac^T; jac 0]
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
      virtual void adaptRowMax(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* hes_tr_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* scaling_vector) = 0;

      /**
       * @brief Diagonally scales the lhs from the left and right, and
       *        diagonally scales rhs from the left
       *
       * @param[in] n_hes - Number of rows in the Hessian matrix.
       * @param[in] n_total - Total number of rows in the system.
       * @param[in,out] hes_i - Row pointers for the Hessian matrix.
       * @param[in,out] hes_j - Column indices for the Hessian matrix.
       * @param[in,out] hes_v - Values for the Hessian matrix.
       * @param[in,out] jac_i - Row pointers for the Jacobian matrix.
       * @param[in,out] jac_j - Column indices for the Jacobian matrix.
       * @param[in,out] jac_v - Values for the Jacobian matrix.
       * @param[in,out] jac_tr_i - Row pointers for the transposed Jacobian matrix.
       * @param[in,out] jac_tr_j - Column indices for the transposed Jacobian matrix.
       * @param[in,out] jac_tr_v - Values for the transposed Jacobian matrix.
       * @param[in,out] rhs1 - Right-hand side vector for the top block.
       * @param[in,out] rhs2 - Right-hand side vector for the bottom block.
       * @param[in,out] aggregate_scaling_vector - Cumulative scaling vector.
       * @param[in] scaling_vector - Scaling vector for the current iteration.
       */
      virtual void adaptDiagScale(index_type n_hes, index_type n_total, index_type* hes_i, index_type* hes_j, real_type* hes_v, index_type* jac_i, index_type* hes_tr_j, real_type* jac_v, index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v, real_type* rhs1, real_type* rhs2, real_type* aggregate_scaling_vector, real_type* scaling_vector) = 0;
    };

  } // namespace hykkt
} // namespace ReSolve
