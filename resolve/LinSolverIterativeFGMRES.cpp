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
#include <limits>
#include <algorithm> // Required for std::max
#include "LinSolverIterativeFGMRES.hpp"
#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/random/SketchingHandler.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/ExampleHelper.hpp>

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
    tol_       = 1e-14;
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



int LinSolverIterativeFGMRES::solve(vector_type* rhs, vector_type* x)
{
  using namespace constants;

  // Initialize the effective stability to 0 for each new solve
  effectiveStability_ = 0.0;

  // io::Logger::setVerbosity(io::Logger::EVERYTHING);

  int outer_flag = 1;
  int notconv    = 1;
  int i          = 0;
  int it         = 0;
  int j          = 0;
  int k          = 0;
  int k1         = 0;

  real_type   t      = 0.0;
  real_type   rnorm  = 0.0;
  real_type   bnorm  = 0.0;
  real_type   rhsnorm = 0.0;
  real_type   tolrel;
  real_type   relnorm = 0.0;

  // Compute the residual
  vec_R_->setToZero(memspace_);
  vec_R_->copyDataFrom(rhs, memspace_, memspace_);
  matrix_handler_->matvec(A_, x, vec_R_, &MINUS_ONE, &ONE, memspace_);

  // Create temporary vectors used inside the solve loop
  vector_type vec_v(n_);
  vector_type vec_z(n_);

  // Vector to hold a copy of v_i for the stability calculation
  vector_type vec_v_copy(n_);
  vec_v_copy.allocate(memspace_);

  // Arnoldi Basis
  vec_Z_->setToZero(memspace_);
  vec_V_->setToZero(memspace_);

  // Initializing Residual and Update
  vec_Y_->setToZero(memspace_);

  // Computing the first Arnodi basis vector
  vec_R_->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
  rnorm = 0.0;
  bnorm = vector_handler_->dot(vec_R_, vec_R_, memspace_);
  rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
  rnorm = std::sqrt(rnorm);

  // Checking if rnorm > norm of RHS
  rhsnorm = vector_handler_->dot(rhs, rhs, memspace_);
  rhsnorm = std::sqrt(rhsnorm);
  if (rnorm > rhsnorm)
      {
	out::warning() << "Initial guess is invalid." << std::endl;
        return 1;
      }

  // rnorm = ||V_1||
  bnorm = std::sqrt(bnorm);
  io::Logger::misc() << "it 0: norm of residual "
                     << std::scientific << std::setprecision(16)
                     << rnorm << " Norm of rhs: " << bnorm << "\n";
  initial_residual_norm_ = rnorm;

  // Outer While
  while (outer_flag)
  {

    if (it == 0)
      {
        tolrel = tol_ * rnorm;
        if (std::abs(tolrel) < MACHINE_EPSILON)
        {
          tolrel = MACHINE_EPSILON;
        }
    }


    bool exit_cond = false;
    switch (conv_cond_)
    {
    case 0:
      exit_cond = ((std::abs(rnorm - ZERO) <= MACHINE_EPSILON));
      break;
    case 1:
      exit_cond = ((std::abs(rnorm - ZERO) <= MACHINE_EPSILON) || (rnorm < tol_));
      break;
    case 2:
      exit_cond = ((std::abs(rnorm - ZERO) <= MACHINE_EPSILON) || (rnorm < (tol_ * bnorm)));
      break;
    }

    if (exit_cond)
    {
      outer_flag         = 0;
      final_residual_norm_   = rnorm;
      initial_residual_norm_ = rnorm;
      total_iters_           = 0;
      break;
    }

    // normalize first vector
    t = 1.0 / rnorm;
    vector_handler_->scal(&t, vec_V_, memspace_);
    // initialize norm history
    h_rs_[0] = rnorm;
    i        = -1;
    notconv  = 1;

    while ((notconv) && (it < maxit_))
    {
      i++;
      it++;

      // Z_i = (LU)^{-1}*V_i
      vec_v.setData(vec_V_->getData(i, memspace_), memspace_);
      if (flexible_)
      {
        vec_z.setData(vec_Z_->getData(i, memspace_), memspace_);
      }
      else
      {
        vec_z.setData(vec_Z_->getData(0, memspace_), memspace_);
      }
      this->precV(&vec_v, &vec_z);
      mem_.deviceSynchronize();

      // orthogonalize V[i+1], form a column of h_H_
      GS_->orthogonalize(n_, vec_V_, h_H_, i);
      if (i != 0)
      {
        for (index_type k = 1; k <= i; k++)
        {
          k1            = k - 1;
          t             = h_H_[i * (restart_ + 1) + k1];
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

      h_H_[(i) * (restart_ + 1) + (i)]      = h_c_[i] * Hii + h_s_[i] * Hii1;
      h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

      // residual norm estimate
      rnorm = std::abs(h_rs_[i + 1]);
      io::Logger::misc() << "it: " << it << " --> norm of the residual "
                         << std::scientific << std::setprecision(16)
                         << rnorm << "\n";
      // check convergence
      if (i + 1 >= restart_ || rnorm <= tolrel || it >= maxit_)
      {
        notconv = 0;
      }
    } // inner while

    io::Logger::misc() << "End of cycle, ESTIMATED norm of residual "
                       << std::scientific << std::setprecision(16)
                       << rnorm << "\n";
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

    // get solution
    if (flexible_)
    {
      for (j = 0; j <= i; j++)
      {
        vec_z.setData(vec_Z_->getData(j, memspace_), memspace_);
        vector_handler_->axpy(&h_rs_[j], &vec_z, vec_Y_, memspace_);
      }
    }
    else
    {
      vec_Z_->setToZero(memspace_);
      vec_z.setData(vec_Z_->getData(0, memspace_), memspace_);
      for (j = 0; j <= i; j++)
      {
        vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
        vector_handler_->axpy(&h_rs_[j], &vec_v, &vec_z, memspace_);
      }
      // now multiply d_Z by precon
      vec_v.setData(vec_V_->getData(memspace_), memspace_);
      this->precV(&vec_z, &vec_v);
      // and add to x
      vector_handler_->axpy(&ONE, &vec_v, vec_Y_, memspace_);
    }

    /* test solution */
    if (rnorm <= tolrel || it >= maxit_)
    {
      // rnorm_aux = rnorm;
      outer_flag = 0;
    }

    vec_R_->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);
    matrix_handler_->matvec(A_, vec_Y_, vec_V_, &MINUS_ONE, &ONE, memspace_);
    rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
    // rnorm = ||V_1||
    rnorm = std::sqrt(rnorm);

    if (!outer_flag)
    {
      final_residual_norm_ = rnorm;
      total_iters_         = it;
      io::Logger::misc() << "End of cycle, solved for update with residual "
                         << std::scientific << std::setprecision(16)
                         << rnorm << "\n";
    }
  } // outer while

  // Compute the norm of residual with the update
  vector_handler_->axpy(&ONE, x, vec_Y_, memspace_);
  vec_R_->copyDataFrom(rhs, memspace_, memspace_);
  matrix_handler_->matvec(A_, vec_Y_, vec_R_, &MINUS_ONE, &ONE, memspace_);
  rnorm = vector_handler_->dot(vec_R_, vec_R_, memspace_);
  // rnorm = ||V_1||
  rnorm = std::sqrt(rnorm);
  relnorm = rnorm / rhsnorm;

  // Compare this with bnorm and update x accordingly
  if (rnorm <= initial_residual_norm_)
      {
	std::cout << "Update to intial guess is successful, final residual (solution plus update) "
	  << std::scientific << std::setprecision(16)
          << relnorm << "\n";

	  x->copyDataFrom(vec_Y_, memspace_, memspace_);
      }
  else {
	std::cout << "Update to intial guess is not successful, final residual greater than initial residual "
          << std::scientific << std::setprecision(16)
          << relnorm << "\n";
       }

  return 0;
}



  int LinSolverIterativeFGMRES::setupPreconditioner(std::string type, LinSolverDirect* LU_solver)
  {
    if (type != "LU")
    {
      out::warning() << "Only LU-type solve can be used as a preconditioner at this time." << std::endl;
      return 1;
    }
    else
    {
      LU_solver_ = LU_solver;
      return 0;
    }
  }

  int LinSolverIterativeFGMRES::resetMatrix(matrix::Sparse* new_matrix)
  {
    A_ = new_matrix;
    matrix_handler_->setValuesChanged(true, memspace_);
    return 0;
  }

  /**
  * @brief Returns the computed effective stability of the preconditioner.
  * @return real_type - The effective stability value.
  */
  real_type LinSolverIterativeFGMRES::getEffectiveStability() const
  {
      return effectiveStability_;
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
    vec_R_ = new vector_type(n_);
    vec_Y_ = new vector_type(n_);
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
    delete vec_R_;
    delete vec_Y_;

    h_H_   = nullptr;
    h_c_   = nullptr;
    h_s_   = nullptr;
    h_rs_  = nullptr;
    vec_V_ = nullptr;
    vec_Z_ = nullptr;
    vec_R_ = nullptr;
    vec_Y_ = nullptr;
    return 0;
  }

  void LinSolverIterativeFGMRES::precV(vector_type* rhs, vector_type* x)
  {
    LU_solver_->solve(rhs, x);
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
