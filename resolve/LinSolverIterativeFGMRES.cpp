/**
 * @file LinSolverIterativeFGMRES.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Implementation of LinSolverIterativeFGMRES class
 *
 */
#include "LinSolverIterativeFGMRES.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <resolve/GramSchmidt.hpp>
#include <resolve/Preconditioner.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/random/SketchingHandler.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  using out = io::Logger;

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES(MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
  {
    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    GS_             = gs;
    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES(index_type     restart,
                                                     real_type      tol,
                                                     index_type     maxit,
                                                     index_type     conv_cond,
                                                     MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
  {
    // Base class settings here (to be removed when solver parameter settings are implemented)
    tol_       = tol;
    maxit_     = maxit;
    restart_   = restart;
    conv_cond_ = conv_cond;
    flexible_  = true;

    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    GS_             = gs;
    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeFGMRES::~LinSolverIterativeFGMRES()
  {
    if (is_solver_set_)
    {
      freeSolverData();
    }
  }

  /**
   * @brief Set pointer to system matrix and allocate solver data.
   *
   * @param[in] A - Sparse system matrix
   *
   * @pre A is a valid sparse matrix
   *
   * @post A_ == A
   * @post Solver data allocated.
   */
  int LinSolverIterativeFGMRES::setup(matrix::Sparse* A)
  {
    // If A_ is already set, then report error and exit.
    if (n_ != A->getNumRows())
    {
      if (is_solver_set_)
      {
        out::warning() << "Matrix size changed. Reallocating solver ...\n";
        freeSolverData();
        is_solver_set_ = false;
      }
    }

    // Set pointer to matrix A and the matrix size.
    A_ = A;
    n_ = A->getNumRows();

    // Allocate solver data.
    if (!is_solver_set_)
    {
      allocateSolverData();
      is_solver_set_ = true;
    }

    GS_->setup(n_, restart_);

    return 0;
  }

  /**
   * @brief Solve linear system A*x = rhs
   *
   * Implements restarted GMRES with optional flexible (FGMRES) variant.
   *
   * Flexible GMRES allows the preconditioner to vary periteration and
   * uses right preconditioning. Standard GMRES supports both left and
   * right preconditioning.
   *
   * Left preconditioning solves M^{-1}Ax = M^{-1}b and checks convergence
   * with ||M^{-1}(b - Ax)||. Right preconditioning solves AM^{-1}(Mx) = b
   * and checks convergence with ||b - Ax||. Both report the true relative
   * residual ||b - Ax||/||b||.
   *
   * @param rhs - right hand side vector
   * @param x   - solution vector
   * @return int - zero if successful, error code otherwise
   *
   * @invariant rhs vector is unchanged.
   * @post x is overwritten with the solution to the linear system.
   */
  int LinSolverIterativeFGMRES::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    if (preconditioner_ == nullptr)
    {
      out::error() << "Preconditioner not set for GMRES solver.\n";
      return 1;
    }

    // Flexible GMRES only supports right preconditioning.
    if (flexible_ && preconditioner_->getSide() == Preconditioner::Side::LEFT)
    {
      out::error() << "Flexible GMRES does not support left preconditioning. "
                   << "Use right preconditioning or disable flexible GMRES.\n";
      return 1;
    }

    // io::Logger::setVerbosity(io::Logger::EVERYTHING);

    int outer_flag = 1;
    int notconv    = 1;
    int i          = 0;
    int it         = 0;
    int j          = 0;
    int k          = 0;
    int k1         = 0;

    real_type   t              = 0.0;
    real_type   res_norm       = 0.0; // Residual norm used for convergence
    real_type   rhs_norm       = 0.0; // Right-hand side norm used for convergence
    real_type   true_res_norm  = 0.0; // True (unpreconditioned) residual norm ||b - Ax|| for reporting
    real_type   true_rhs_norm  = 0.0; // True (unpreconditioned) right-hand side norm ||b|| for reporting
    real_type   initial_r_norm = 0.0;
    real_type   final_r_norm   = 0.0;
    real_type   x_norm         = 0.0;
    real_type   tol_rel;
    vector_type vec_v(n_);
    vector_type vec_z(n_);
    vector_type x_initial(x->getSize());

    x_initial.allocate(memspace_);
    x_initial.copyFromExternal(x, memspace_, memspace_);

    x_norm = vector_handler_->dot(&x_initial, &x_initial, memspace_);
    x_norm = std::sqrt(x_norm);

    bool check_initial_guess = (x_norm > MACHINE_EPSILON);

    // Compute initial residual norm.
    // V[0] = ||b - A*x0||         for right preconditioning
    // V[0] = ||M^{-1}{b - A*x0}|| for left preconditioning

    vec_Z_->setToZero(memspace_);
    vec_V_->setToZero(memspace_);

    rhs->copyToExternal(vec_V_->getData(memspace_), 0, memspace_, memspace_);
    matrix_handler_->matvec(A_, x, vec_V_, &MINUS_ONE, &ONE, memspace_);

    vec_v.setData(vec_V_->getData(0, memspace_), memspace_);
    vec_z.setData(vec_Z_->getData(0, memspace_), memspace_);

    // True residual norm ||b - A*x0||
    true_res_norm = vector_handler_->dot(&vec_v, &vec_v, memspace_);
    true_res_norm = std::sqrt(true_res_norm);

    // True right-hand side norm ||b||
    true_rhs_norm = vector_handler_->dot(rhs, rhs, memspace_);
    true_rhs_norm = std::sqrt(true_rhs_norm);

    initial_r_norm = true_res_norm;

    if (check_initial_guess && initial_r_norm > true_rhs_norm)
    {
      out::warning() << "Initial guess has a larger residual than the zero vector. Ignoring initial guess.\n";

      x->setToZero(memspace_);
      x_initial.copyFromExternal(x, memspace_, memspace_);
      check_initial_guess = false;

      vec_Z_->setToZero(memspace_);
      vec_V_->setToZero(memspace_);

      rhs->copyToExternal(vec_V_->getData(memspace_), 0, memspace_, memspace_);
      matrix_handler_->matvec(A_, x, vec_V_, &MINUS_ONE, &ONE, memspace_);

      vec_v.setData(vec_V_->getData(0, memspace_), memspace_);

      true_res_norm  = vector_handler_->dot(&vec_v, &vec_v, memspace_);
      true_res_norm  = std::sqrt(true_res_norm);
      initial_r_norm = true_res_norm;
    }

    switch (preconditioner_->getSide())
    {
    case Preconditioner::Side::RIGHT:
      // Right preconditioning uses true norms for convergence
      res_norm = true_res_norm;
      rhs_norm = true_rhs_norm;
      break;
    case Preconditioner::Side::LEFT:
      // Left preconditioning uses preconditioned norms for convergence
      // Left-preconditioned residual norm ||M^{-1}*(b-A*x0)||
      preconditioner_->apply(&vec_v, &vec_z);
      vec_v.copyFromExternal(&vec_z, memspace_, memspace_);
      res_norm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
      res_norm = std::sqrt(res_norm);

      // Left-preconditioned right-hand side norm ||M^{-1}*b||
      preconditioner_->apply(rhs, &vec_z);
      rhs_norm = vector_handler_->dot(&vec_z, &vec_z, memspace_);
      rhs_norm = std::sqrt(rhs_norm);
      break;
    default:
      out::error() << "Unknown preconditioner side.\n";
      return 1;
    }

    io::Logger::misc() << "it 0: norm of residual "
                       << std::scientific << std::setprecision(16)
                       << res_norm << " Norm of rhs: " << rhs_norm << "\n";

    // Report the true initial relative residual norm
    initial_residual_norm_ = true_res_norm / true_rhs_norm;

    while (outer_flag)
    {
      if (it == 0)
      {
        tol_rel = tol_ * res_norm;
        if (std::abs(tol_rel) < MACHINE_EPSILON)
        {
          tol_rel = MACHINE_EPSILON;
        }
      }

      bool exit_cond = false;
      switch (conv_cond_)
      {
      case 0:
        exit_cond = ((std::abs(res_norm - ZERO) <= MACHINE_EPSILON));
        break;
      case 1:
        exit_cond = ((std::abs(res_norm - ZERO) <= MACHINE_EPSILON) || (res_norm < tol_));
        break;
      case 2:
        exit_cond = ((std::abs(res_norm - ZERO) <= MACHINE_EPSILON) || (res_norm < (tol_ * rhs_norm)));
        break;
      }

      if (exit_cond)
      {
        outer_flag             = 0;
        final_residual_norm_   = res_norm;
        initial_residual_norm_ = res_norm;
        total_iters_           = 0;
        break;
      }

      // normalize first vector
      t = 1.0 / res_norm;
      vector_handler_->scal(t, vec_V_, memspace_);
      // initialize norm history
      h_rs_[0] = res_norm;
      i        = -1;
      notconv  = 1;

      while ((notconv) && (it < maxit_))
      {
        i++;
        it++;

        vec_v.setData(vec_V_->getData(i, memspace_), memspace_);
        if (flexible_)
        {
          vec_z.setData(vec_Z_->getData(i, memspace_), memspace_);
        }
        else
        {
          vec_z.setData(vec_Z_->getData(0, memspace_), memspace_);
        }

        // Expand the Krylov subspace.
        //
        // New basis vector:
        //   V[i+1] = A*M^{-1}*V[i] (right preconditioning)
        //   V[i+1] = M^{-1}*A*V[i] (left preconditioning)

        switch (preconditioner_->getSide())
        {
        case Preconditioner::Side::RIGHT:
          // Compute vec_z = M^{-1}*V[i], then V[i+1] = A*vec_z
          preconditioner_->apply(&vec_v, &vec_z);

          vec_v.setData(vec_V_->getData(i + 1, memspace_), memspace_);
          matrix_handler_->matvec(A_, &vec_z, &vec_v, &ONE, &ZERO, memspace_);
          break;
        case Preconditioner::Side::LEFT:
          // Compute vec_z = A*V[i], then V[i+1] = M^{-1}*vec_z
          matrix_handler_->matvec(A_, &vec_v, &vec_z, &ONE, &ZERO, memspace_);

          vec_v.setData(vec_V_->getData(i + 1, memspace_), memspace_);
          preconditioner_->apply(&vec_z, &vec_v);
          break;
        default:
          out::error() << "Unknown preconditioner side.\n";
          return 1;
        }

        // orthogonalize V[i+1], form a column of h_H_

        GS_->orthogonalize(n_, vec_V_, h_H_, i);
        if (i != 0)
        {
          for (index_type k = 1; k <= i; k++)
          {
            k1                            = k - 1;
            t                             = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k]  = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if i!=0
        real_type Hii  = h_H_[i * (restart_ + 1) + i];
        real_type Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
        real_type gam  = std::sqrt(Hii * Hii + Hii1 * Hii1);

        if (std::abs(gam - ZERO) <= MACHINE_EPSILON)
        {
          gam = MACHINE_EPSILON;
        }

        /* next Given's rotation */
        h_c_[i]      = Hii / gam;
        h_s_[i]      = Hii1 / gam;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i]     = h_c_[i] * h_rs_[i];

        h_H_[(i) * (restart_ + 1) + (i)]     = h_c_[i] * Hii + h_s_[i] * Hii1;
        h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

        // residual norm estimate
        res_norm = std::abs(h_rs_[i + 1]);
        io::Logger::misc() << "it: " << it << " --> norm of the residual "
                           << std::scientific << std::setprecision(16)
                           << res_norm << "\n";
        // check convergence
        if (i + 1 >= restart_ || res_norm <= tol_rel || it >= maxit_)
        {
          notconv = 0;
        }
      } // inner while

      io::Logger::misc() << "End of cycle, ESTIMATED norm of residual "
                         << std::scientific << std::setprecision(16)
                         << res_norm << "\n";
      // solve tri system
      h_rs_[i] = h_rs_[i] / h_H_[i * (restart_ + 1) + i];
      for (int ii = 2; ii <= i + 1; ii++)
      {
        k  = i - ii + 1;
        k1 = k + 1;
        t  = h_rs_[k];
        for (j = k1; j <= i; j++)
        {
          t -= h_H_[j * (restart_ + 1) + k] * h_rs_[j];
        }
        h_rs_[k] = t / h_H_[k * (restart_ + 1) + k];
      }

      // Update the approximate solution x using h_rs_.
      // Flexible GMRES uses the preconditioned basis Z[j] directly.
      // Standard GMRES first forms vec_z from V[j], then applies M^{-1}
      // only for right preconditioning.

      if (flexible_)
      {
        for (j = 0; j <= i; j++)
        {
          vec_z.setData(vec_Z_->getData(j, memspace_), memspace_);
          vector_handler_->axpy(h_rs_[j], &vec_z, x, memspace_);
        }
      }
      else
      {
        // Accumulate the correction vec_z = sum_j h_rs_[j] * V[j]
        vec_Z_->setToZero(memspace_);
        vec_z.setData(vec_Z_->getData(0, memspace_), memspace_);
        for (j = 0; j <= i; j++)
        {
          vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
          vector_handler_->axpy(h_rs_[j], &vec_v, &vec_z, memspace_);
        }
        // Apply the correction to x based on preconditioning side
        switch (preconditioner_->getSide())
        {
        case Preconditioner::Side::RIGHT:
          // Right preconditioning: x += M^{-1} * vec_z
          preconditioner_->apply(&vec_z, &vec_v);
          vector_handler_->axpy(ONE, &vec_v, x, memspace_);
          break;
        case Preconditioner::Side::LEFT:
          // Left preconditioning: x += vec_z
          vector_handler_->axpy(ONE, &vec_z, x, memspace_);
          break;
        default:
          out::error() << "Unknown preconditioner side.\n";
          return 1;
        }
      }

      /* test solution */

      if (res_norm <= tol_rel || it >= maxit_)
      {
        // res_norm_aux = res_norm;
        outer_flag = 0;
      }

      rhs->copyToExternal(vec_V_->getData(memspace_), 0, memspace_, memspace_);
      matrix_handler_->matvec(A_, x, vec_V_, &MINUS_ONE, &ONE, memspace_);

      vec_v.setData(vec_V_->getData(0, memspace_), memspace_);
      true_res_norm = vector_handler_->dot(&vec_v, &vec_v, memspace_);
      true_res_norm = std::sqrt(true_res_norm);

      // Left-preconditioned GMRES applies M^{-1} to the residual
      if (preconditioner_->getSide() == Preconditioner::Side::LEFT)
      {
        preconditioner_->apply(&vec_v, &vec_z);
        vec_v.copyFromExternal(&vec_z, memspace_, memspace_);
      }
      res_norm = vector_handler_->dot(vec_V_, vec_V_, memspace_);

      // res_norm = ||V_1||
      res_norm = std::sqrt(res_norm);

      if (!outer_flag)
      {
        final_r_norm = true_res_norm;

        if (check_initial_guess && final_r_norm > initial_r_norm)
        {
          out::warning() << "Iterative solver did not improve the initial guess. Returning the initial guess.\n";
          x->copyFromExternal(&x_initial, memspace_, memspace_);
          final_r_norm  = initial_r_norm;
          true_res_norm = final_r_norm;
        }

        // Report the true relative residual norm
        final_residual_norm_ = true_res_norm / true_rhs_norm;
        total_iters_         = it;
        io::Logger::misc() << "End of cycle, COMPUTED norm of residual "
                           << std::scientific << std::setprecision(16)
                           << res_norm << "\n";
      }
    } // outer while

    return 0;
  }

  int LinSolverIterativeFGMRES::resetMatrix(matrix::Sparse* new_matrix)
  {
    A_ = new_matrix;
    matrix_handler_->setValuesChanged(true, memspace_);
    return 0;
  }

  /**
   * @brief Sets pointer to Gram-Schmidt (re)orthogonalization.
   *
   * @param[in] gs - pointer to Gram-Schmidt class instance.
   * @return 0 if successful, error code otherwise.
   */
  int LinSolverIterativeFGMRES::setOrthogonalization(GramSchmidt* gs)
  {
    GS_ = gs;
    return 0;
  }

  /**
   * @brief Set/change GMRES restart value
   *
   * This function should leave solver instance in the same state but with
   * the new restart value.
   *
   * @param[in] restart - the restart value
   * @return 0 if successful, error code otherwise.
   *
   * @todo Consider not setting up GS, if it was not previously set up.
   */
  int LinSolverIterativeFGMRES::setRestart(index_type restart)
  {
    // If the new restart value is the same as the old, do nothing.
    if (restart_ == restart)
    {
      return 0;
    }

    // Otherwise, set new restart value
    restart_ = restart;

    // If solver is already set, reallocate solver data
    if (is_solver_set_)
    {
      freeSolverData();
      allocateSolverData();
    }

    matrix_handler_->setValuesChanged(true, memspace_);

    // If Gram-Schmidt is already set, we need to reallocate it since the
    // restart value has changed.
    // if (GS_->isSetupComplete()) {
    GS_->setup(n_, restart_);
    // }

    return 0;
  }

  /**
   * @brief Switches between flexible and standard GMRES
   *
   * @param is_flexible - true means set flexible GMRES
   * @return 0 if successful, error code otherwise.
   */
  int LinSolverIterativeFGMRES::setFlexible(bool is_flexible)
  {
    // TODO: Add vector method resize
    if (vec_Z_)
    {
      delete vec_Z_;
      if (is_flexible)
      {
        vec_Z_ = new vector_type(n_, restart_ + 1);
      }
      else
      {
        // otherwise Z is just a one vector, not multivector and we dont keep it
        vec_Z_ = new vector_type(n_);
      }
      vec_Z_->allocate(memspace_);
    }
    flexible_ = is_flexible;
    matrix_handler_->setValuesChanged(true, memspace_);
    return 0;
  }

  /**
   * @brief Set the convergence condition for GMRES solver
   *
   * @param[in] conv_cond - Possible values: 0, 1, 2
   * @return int - error code, 0 if successful
   */
  int LinSolverIterativeFGMRES::setConvergenceCondition(index_type conv_cond)
  {
    conv_cond_ = conv_cond;
    return 0;
  }

  index_type LinSolverIterativeFGMRES::getRestart() const
  {
    return restart_;
  }

  index_type LinSolverIterativeFGMRES::getConvCond() const
  {
    return conv_cond_;
  }

  bool LinSolverIterativeFGMRES::getFlexible() const
  {
    return flexible_;
  }

  int LinSolverIterativeFGMRES::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
    case TOL:
      setTol(atof(value.c_str()));
      break;
    case MAXIT:
      setMaxit(atoi(value.c_str()));
      break;
    case RESTART:
      setRestart(atoi(value.c_str()));
      break;
    case CONV_COND:
      setConvergenceCondition(atoi(value.c_str()));
      break;
    case FLEXIBLE:
      setFlexible(value == "yes");
      break;
    default:
      std::cout << "Setting parameter failed!\n";
    }
    return 0;
  }

  std::string LinSolverIterativeFGMRES::getCliParamString(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown string parameter " << id << "\n";
    }
    return "";
  }

  index_type LinSolverIterativeFGMRES::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
    case MAXIT:
      return getMaxit();
      break;
    case RESTART:
      return getRestart();
      break;
    case CONV_COND:
      return getConvCond();
      break;
    default:
      out::error() << "Trying to get unknown integer parameter " << id << "\n";
    }
    return -1;
  }

  real_type LinSolverIterativeFGMRES::getCliParamReal(const std::string id) const
  {
    switch (getParamId(id))
    {
    case TOL:
      return getTol();
      break;
    default:
      out::error() << "Trying to get unknown real parameter " << id << "\n";
    }
    return std::numeric_limits<real_type>::quiet_NaN();
  }

  bool LinSolverIterativeFGMRES::getCliParamBool(const std::string id) const
  {
    switch (getParamId(id))
    {
    case FLEXIBLE:
      return getFlexible();
      break;
    default:
      out::error() << "Trying to get unknown boolean parameter " << id << "\n";
    }
    return false;
  }

  int LinSolverIterativeFGMRES::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    case TOL:
      std::cout << getTol() << "\n";
      break;
    case MAXIT:
      std::cout << getMaxit() << "\n";
      break;
    case RESTART:
      std::cout << getRestart() << "\n";
      break;
    case CONV_COND:
      std::cout << getConvCond() << "\n";
      break;
    case FLEXIBLE:
      std::cout << getFlexible() << "\n";
      break;
    default:
      out::error() << "Trying to print unknown parameter " << id << "\n";
      return 1;
    }
    return 0;
  }

  //
  // Private methods
  //

  int LinSolverIterativeFGMRES::allocateSolverData()
  {
    vec_V_ = new vector_type(n_, restart_ + 1);
    vec_V_->allocate(memspace_);
    if (flexible_)
    {
      vec_Z_ = new vector_type(n_, restart_ + 1);
    }
    else
    {
      // otherwise Z is just a one vector, not multivector and we dont keep it
      vec_Z_ = new vector_type(n_);
    }
    vec_Z_->allocate(memspace_);
    h_H_  = new real_type[restart_ * (restart_ + 1)];
    h_c_  = new real_type[restart_];     // needed for givens
    h_s_  = new real_type[restart_];     // same
    h_rs_ = new real_type[restart_ + 1]; // for residual norm history

    return 0;
  }

  int LinSolverIterativeFGMRES::freeSolverData()
  {
    delete[] h_H_;
    delete[] h_c_;
    delete[] h_s_;
    delete[] h_rs_;
    delete vec_V_;
    delete vec_Z_;

    h_H_   = nullptr;
    h_c_   = nullptr;
    h_s_   = nullptr;
    h_rs_  = nullptr;
    vec_V_ = nullptr;
    vec_Z_ = nullptr;

    return 0;
  }

  void LinSolverIterativeFGMRES::setMemorySpace()
  {
    bool is_matrix_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_matrix_handler_hip  = matrix_handler_->getIsHipEnabled();
    bool is_vector_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_vector_handler_hip  = matrix_handler_->getIsHipEnabled();

    if ((is_matrix_handler_cuda != is_vector_handler_cuda) || (is_matrix_handler_hip != is_vector_handler_hip))
    {
      out::error() << "Matrix and vector handler backends are incompatible!\n";
    }

    if (is_matrix_handler_cuda || is_matrix_handler_hip)
    {
      memspace_ = memory::DEVICE;
    }
    else
    {
      memspace_ = memory::HOST;
    }
  }

  void LinSolverIterativeFGMRES::initParamList()
  {
    params_list_["tol"]       = TOL;
    params_list_["maxit"]     = MAXIT;
    params_list_["restart"]   = RESTART;
    params_list_["conv_cond"] = CONV_COND;
    params_list_["flexible"]  = FLEXIBLE;
  }

} // namespace ReSolve
