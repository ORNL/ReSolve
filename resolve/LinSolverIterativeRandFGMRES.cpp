/**
 * @file LinSolverIterativeRandFGMRES.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Implementation of LinSolverIterativeRandFGMRES class.
 *
 */
#include "LinSolverIterativeRandFGMRES.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
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

  LinSolverIterativeRandFGMRES::LinSolverIterativeRandFGMRES(MatrixHandler*  matrix_handler,
                                                             VectorHandler*  vector_handler,
                                                             SketchingMethod rand_method,
                                                             GramSchmidt*    gs)
  {
    tol_       = 1e-14; // default
    maxit_     = 100;   // default
    restart_   = 10;
    conv_cond_ = 2; // default
    flexible_  = true;

    matrix_handler_   = matrix_handler;
    vector_handler_   = vector_handler;
    sketching_method_ = rand_method;
    GS_               = gs;

    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeRandFGMRES::LinSolverIterativeRandFGMRES(index_type      restart,
                                                             real_type       tol,
                                                             index_type      maxit,
                                                             index_type      conv_cond,
                                                             MatrixHandler*  matrix_handler,
                                                             VectorHandler*  vector_handler,
                                                             SketchingMethod rand_method,
                                                             GramSchmidt*    gs)
  {
    tol_       = tol;
    maxit_     = maxit;
    restart_   = restart;
    conv_cond_ = conv_cond;
    flexible_  = true;

    matrix_handler_   = matrix_handler;
    vector_handler_   = vector_handler;
    sketching_method_ = rand_method;
    GS_               = gs;

    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeRandFGMRES::~LinSolverIterativeRandFGMRES()
  {
    if (is_solver_set_)
    {
      freeSolverData();
    }

    if (is_sketching_set_)
    {
      freeSketchingData();
    }
  }

  /**
   * @brief Set system matrix and allocate solver and sketching data
   *
   * @param[in] A - sparse system matrix
   * @return 0 if setup successful
   */
  int LinSolverIterativeRandFGMRES::setup(matrix::Sparse* A)
  {
    // If A_ is already set, then report error and exit.
    if (n_ != A->getNumRows())
    {
      if (is_solver_set_)
      {
        out::warning() << "Matrix size changed. Reallocating solver ...\n";
        freeSolverData();
      }

      if (is_sketching_set_)
      {
        out::warning() << "Matrix size changed. Reallocating solver ...\n";
        freeSketchingData();
      }
    }

    A_ = A;
    n_ = A_->getNumRows();

    if (!is_solver_set_)
    {
      allocateSolverData();
    }

    if (!is_sketching_set_)
    {
      allocateSketchingData();
    }

    GS_->setup(k_rand_, restart_);

    return 0;
  }

  /**
   * @brief Solve linear system A * x = rhs
   *
   * Implements restarted randomized GMRES with optional flexible
   * variant. A sketching operator is used to orthogonalize in a reduced space
   *
   * Flexible randomized GMRES allows the preconditioner to vary per
   * iteration and uses right preconditioning. Standard randomized GMRES
   * supports both left and right preconditioning.
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
  int LinSolverIterativeRandFGMRES::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    if (preconditioner_ == nullptr)
    {
      out::error() << "Preconditioner not set for randomized GMRES solver.\n";
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

    int        outer_flag = 1;
    int        notconv    = 1;
    index_type i          = 0;
    int        it         = 0;
    int        j;
    int        k;
    int        k1;

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
    vector_type vec_s(k_rand_);
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

    // Left preconditioning uses preconditioned norms for convergence
    if (preconditioner_->getSide() == Preconditioner::LEFT)
    {
      preconditioner_->apply(&vec_v, &vec_z);
      vec_v.copyFromExternal(&vec_z, memspace_, memspace_);
    }

    vec_s.setData(vec_S_->getData(0, memspace_), memspace_);

    sketching_handler_->Theta(&vec_v, &vec_s);

    if (sketching_method_ == fwht)
    {
      vector_handler_->scal(one_over_k_, &vec_s, memspace_);
    }
    res_norm = vector_handler_->dot(&vec_s, &vec_s, memspace_);
    rhs_norm = vector_handler_->dot(rhs, rhs, memspace_);

    // Left-preconditioned right-hand side norm ||M^{-1}*b||
    if (!flexible_ && preconditioner_->getSide() == Preconditioner::LEFT)
    {
      preconditioner_->apply(rhs, &vec_z);
      rhs_norm = vector_handler_->dot(&vec_z, &vec_z, memspace_);
    }

    res_norm = std::sqrt(res_norm); // res_norm = ||S_0||
    rhs_norm = std::sqrt(rhs_norm);

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
      vector_handler_->scal(t, vec_S_, memspace_);

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
        // this is where it differs from normal solver GS
        vec_s.setData(vec_S_->getData(i + 1, memspace_), memspace_);
        sketching_handler_->Theta(&vec_v, &vec_s);
        if (sketching_method_ == fwht)
        {
          vector_handler_->scal(one_over_k_, &vec_s, memspace_);
        }

        GS_->orthogonalize(k_rand_, vec_S_, h_H_, i);

        // now post-process
        vec_aux_->resize(i + 1);
        vec_aux_->copyFromExternal(&h_H_[i * (restart_ + 1)], memory::HOST, memspace_);

        // V(:, i+1) = w - V(:, 1:i)*d_H_col = V(:, i+1) - d_H_col*V(:,1:i);
        vector_handler_->gemv('N', i + 1, MINUS_ONE, ONE, vec_V_, vec_aux_, &vec_v, memspace_);

        t = 1.0 / h_H_[i * (restart_ + 1) + i + 1];
        vector_handler_->scal(t, &vec_v, memspace_);
        vec_s.setData(vec_S_->getData(i + 1, memspace_), memspace_);

        if (i != 0)
        {
          for (int k = 1; k <= i; k++)
          {
            k1                            = k - 1;
            t                             = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k]  = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if (i != 0)
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
        outer_flag = 0;
      }

      rhs->copyToExternal(vec_V_->getData(memspace_), 0, memspace_, memspace_);
      matrix_handler_->matvec(A_, x, vec_V_, &MINUS_ONE, &ONE, memspace_);
      if (outer_flag)
      {
        // Left-preconditioned GMRES applies M^{-1} to the residual before sketching
        if (!flexible_ && preconditioner_->getSide() == Preconditioner::LEFT)
        {
          vec_v.setData(vec_V_->getData(0, memspace_), memspace_);
          preconditioner_->apply(&vec_v, &vec_z);
          vec_v.copyFromExternal(&vec_z, memspace_, memspace_);
        }

        sketching_handler_->reset();

        if (sketching_method_ == cs)
        {
          vec_S_->setToZero(memspace_);
        }
        vec_v.setData(vec_V_->getData(0, memspace_), memspace_);
        vec_s.setData(vec_S_->getData(0, memspace_), memspace_);
        sketching_handler_->Theta(&vec_v, &vec_s);
        if (sketching_method_ == fwht)
        {
          vector_handler_->scal(one_over_k_, &vec_s, memspace_);
        }
        res_norm = vector_handler_->dot(vec_S_, vec_S_, memspace_);
        // res_norm = ||S_0||
        res_norm = std::sqrt(res_norm);
      }

      if (!outer_flag)
      {
        // Compute the true residual norm = ||b - A*x||
        true_res_norm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
        true_res_norm = std::sqrt(true_res_norm);
        final_r_norm  = true_res_norm;

        if (check_initial_guess && final_r_norm > initial_r_norm)
        {
          out::warning() << "Iterative solver did not improve the initial guess. Returning the initial guess.\n";
          x->copyFromExternal(&x_initial, memspace_, memspace_);
          final_r_norm  = initial_r_norm;
          true_res_norm = final_r_norm;
        }

        io::Logger::misc() << "End of cycle, COMPUTED norm of residual "
                           << std::scientific << std::setprecision(16)
                           << true_res_norm << "\n";

        // Report the true relative residual norm
        final_residual_norm_ = true_res_norm / true_rhs_norm;
        total_iters_         = it;
      }
    } // outer while
    return 0;
  }

  index_type LinSolverIterativeRandFGMRES::getKrand()
  {
    return k_rand_;
  }

  int LinSolverIterativeRandFGMRES::resetMatrix(matrix::Sparse* new_matrix)
  {
    A_ = new_matrix;
    matrix_handler_->setValuesChanged(true, memspace_);
    return 0;
  }

  /**
   * @brief Set sketching method based on input string.
   *
   * @param[in] method - string describing sketching method
   * @return 0 if successful, 1 otherwise.
   */
  int LinSolverIterativeRandFGMRES::setSketchingMethod(SketchingMethod method)
  {
    if (is_sketching_set_)
    {
      if (method == sketching_method_)
      {
        out::misc() << "Keeping sketching method " << method << "\n";
        return 0;
      }
      out::misc() << "Deleting sketching method " << sketching_method_ << "\n";
      freeSketchingData();
    }

    // If solver is set, go ahead and create sketching, otherwise just set sketching method.
    sketching_method_ = method;
    if (is_solver_set_)
    {
      out::misc() << "Allocating sketching method " << sketching_method_ << "\n";
      allocateSketchingData();
    }

    // If Gram-Schmidt is already set, we need to reallocate it since the
    // k_rand_ value has changed.
    GS_->setup(k_rand_, restart_);
    matrix_handler_->setValuesChanged(true, memspace_);

    return 0;
  }

  int LinSolverIterativeRandFGMRES::setOrthogonalization(GramSchmidt* gs)
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
  int LinSolverIterativeRandFGMRES::setRestart(index_type restart)
  {
    // If the new restart value is the same as the old, do nothing.
    if (restart_ == restart)
    {
      return 0;
    }

    // Set new restart value
    restart_ = restart;

    // If solver is already set, reallocate solver data
    if (is_solver_set_)
    {
      freeSolverData();
      allocateSolverData();
    }

    // If sketching has been set, reallocate sketching data
    if (is_sketching_set_)
    {
      freeSketchingData();
      allocateSketchingData();
    }

    // If Gram-Schmidt is already set, we need to reallocate it since the
    // restart value has changed.
    GS_->setup(k_rand_, restart_);
    matrix_handler_->setValuesChanged(true, memspace_);

    return 0;
  }

  /**
   * @brief Switches between flexible and standard GMRES
   *
   * @param is_flexible - true means set flexible GMRES
   * @return 0 if successful, error code otherwise.
   */
  int LinSolverIterativeRandFGMRES::setFlexible(bool is_flexible)
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
        // otherwise Z is just one vector, not a multivector and we don't keep it
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
  int LinSolverIterativeRandFGMRES::setConvergenceCondition(index_type conv_cond)
  {
    conv_cond_ = conv_cond;
    return 0;
  }

  index_type LinSolverIterativeRandFGMRES::getRestart() const
  {
    return restart_;
  }

  index_type LinSolverIterativeRandFGMRES::getConvCond() const
  {
    return conv_cond_;
  }

  bool LinSolverIterativeRandFGMRES::getFlexible() const
  {
    return flexible_;
  }

  int LinSolverIterativeRandFGMRES::setCliParam(const std::string id, const std::string value)
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

  std::string LinSolverIterativeRandFGMRES::getCliParamString(const std::string id) const
  {
    switch (getParamId(id))
    {
    default:
      out::error() << "Trying to get unknown string parameter " << id << "\n";
    }
    return "";
  }

  index_type LinSolverIterativeRandFGMRES::getCliParamInt(const std::string id) const
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

  real_type LinSolverIterativeRandFGMRES::getCliParamReal(const std::string id) const
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

  bool LinSolverIterativeRandFGMRES::getCliParamBool(const std::string id) const
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

  int LinSolverIterativeRandFGMRES::printCliParam(const std::string id) const
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
    default:
      out::error() << "Trying to print unknown parameter " << id << "\n";
      return 1;
    }
    return 0;
  }

  //
  // Private methods
  //

  int LinSolverIterativeRandFGMRES::allocateSolverData()
  {
    vec_V_ = new vector_type(n_, restart_ + 1);
    vec_V_->allocate(memspace_);
    if (flexible_)
    {
      vec_Z_ = new vector_type(n_, restart_ + 1);
    }
    else
    {
      // otherwise Z is just one vector, not multivector and we dont keep it
      vec_Z_ = new vector_type(n_);
    }
    vec_Z_->allocate(memspace_);
    vec_aux_ = new vector_type(restart_ + 1);
    vec_aux_->allocate(memspace_);

    h_H_  = new real_type[restart_ * (restart_ + 1)];
    h_c_  = new real_type[restart_];     // needed for givens
    h_s_  = new real_type[restart_];     // same
    h_rs_ = new real_type[restart_ + 1]; // for residual norm history

    is_solver_set_ = true;
    return 0;
  }

  int LinSolverIterativeRandFGMRES::freeSolverData()
  {
    delete[] h_H_;
    delete[] h_c_;
    delete[] h_s_;
    delete[] h_rs_;
    delete vec_V_;
    delete vec_Z_;
    delete vec_aux_;

    h_H_     = nullptr;
    h_c_     = nullptr;
    h_s_     = nullptr;
    h_rs_    = nullptr;
    vec_V_   = nullptr;
    vec_Z_   = nullptr;
    vec_aux_ = nullptr;

    is_solver_set_ = false;
    return 0;
  }

  /**
   * @brief Allocate data and instantiate sketching handler.
   *
   * @pre Randomized solver data is allocated.
   */
  int LinSolverIterativeRandFGMRES::allocateSketchingData()
  {
    // Set randomized method
    k_rand_ = n_;
    switch (sketching_method_)
    {
    case cs:
      if (std::ceil(restart_ * std::log(n_)) < k_rand_)
      {
        k_rand_ = static_cast<index_type>(std::ceil(restart_ * std::log(static_cast<real_type>(n_))));
      }
      sketching_handler_ = new SketchingHandler(sketching_method_, device_type_);
      // set k and n
      break;
    case fwht:
      if (std::ceil(2.0 * restart_ * std::log(n_) / std::log(restart_)) < k_rand_)
      {
        k_rand_ = static_cast<index_type>(std::ceil(2.0 * restart_ * std::log(n_) / std::log(restart_)));
      }
      sketching_handler_ = new SketchingHandler(sketching_method_, device_type_);
      break;
    default:
      io::Logger::warning() << "Wrong sketching method, setting to default (CountSketch)\n";
      sketching_method_ = cs;
      if (std::ceil(restart_ * std::log(n_)) < k_rand_)
      {
        k_rand_ = static_cast<index_type>(std::ceil(restart_ * std::log(n_)));
      }
      sketching_handler_ = new SketchingHandler(cs, device_type_);
      break;
    }

    one_over_k_ = 1.0 / std::sqrt((real_type) k_rand_);
    vec_S_      = new vector_type(k_rand_, restart_ + 1);
    vec_S_->allocate(memspace_);
    if (sketching_method_ == cs)
    {
      vec_S_->setToZero(memspace_);
    }

    sketching_handler_->setup(n_, k_rand_);
    is_sketching_set_ = true;
    return 0;
  }

  int LinSolverIterativeRandFGMRES::freeSketchingData()
  {
    delete vec_S_;
    delete sketching_handler_;

    vec_S_             = nullptr;
    sketching_handler_ = nullptr;

    is_sketching_set_ = false;
    return 0;
  }

  /**
   * @brief Set memory space and device tape based on how MatrixHandler
   * and VectorHandler are configured.
   *
   */
  void LinSolverIterativeRandFGMRES::setMemorySpace()
  {
    bool is_matrix_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_matrix_handler_hip  = matrix_handler_->getIsHipEnabled();
    bool is_vector_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_vector_handler_hip  = matrix_handler_->getIsHipEnabled();

    if ((is_matrix_handler_cuda != is_vector_handler_cuda) || (is_matrix_handler_hip != is_vector_handler_hip))
    {
      out::error() << "Matrix and vector handler backends are incompatible!\n";
    }

    if (is_matrix_handler_cuda)
    {
      memspace_    = memory::DEVICE;
      device_type_ = memory::CUDADEVICE;
    }
    else if (is_matrix_handler_hip)
    {
      memspace_    = memory::DEVICE;
      device_type_ = memory::HIPDEVICE;
    }
    else
    {
      memspace_    = memory::HOST;
      device_type_ = memory::NONE;
    }
  }

  void LinSolverIterativeRandFGMRES::initParamList()
  {
    params_list_["tol"]       = TOL;
    params_list_["maxit"]     = MAXIT;
    params_list_["restart"]   = RESTART;
    params_list_["conv_cond"] = CONV_COND;
    params_list_["flexible"]  = FLEXIBLE;
  }

} // namespace ReSolve
