#pragma once

#include <cmath>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

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
       * @param[in] n_total - Total number of rows in the 2x2 block system.
       * @param[in] hes - Pointer to the Hessian matrix in CSR format.
       * @param[in] jac - Pointer to the Jacobian matrix in CSR format.
       * @param[in] jac_tr - Pointer to the transposed Jacobian matrix in CSR format.
       * @param[out] scaling_vector - Scaling vector to be updated in-place.
       */
      virtual void adaptRowMax(index_type n_hes, index_type n_total, matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr, vector::Vector* scaling_vector) = 0;

      /**
       * @brief Apply the scaling stored in scaling_vector to the matrices and rhs
       *
       * @post The left hand side is scaled from the left and right and the right hand side
       *       is scaled from the right by the scaling_vector.
       *
       * @param[in] n_hes - Number of rows in the Hessian matrix.
       * @param[in] n_total - Total number of rows in the 2x2 block system.
       * @param[in] hes - Pointer to the Hessian matrix in CSR format to be updated in-place.
       * @param[in] jac - Pointer to the Jacobian matrix in CSR format to be updated in-place.
       * @param[in] jac_tr - Pointer to the transposed Jacobian matrix in CSR format to be updated in-place.
       * @param[in] rhs_top - Right-hand side vector for the top block to be updated in-place.
       * @param[in] rhs_bottom - Right-hand side vector for the bottom block to be updated in-place.
       * @param[in,out] aggregate_scaling_vector - Cumulative scaling vector to be updated in-place.
       * @param[in] scaling_vector - Scaling vector for the current iteration.
       */
      virtual void adaptDiagScale(index_type n_hes, index_type n_total, matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr, vector::Vector* rhs_top, vector::Vector* rhs_bottom, vector::Vector* aggregate_scaling_vector, vector::Vector* scaling_vector) = 0;
    };

  } // namespace hykkt
} // namespace ReSolve
