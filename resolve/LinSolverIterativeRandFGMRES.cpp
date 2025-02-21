/**
 * @file LinSolverIterativeRandFGMRES.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Implementation of LinSolverIterativeRandFGMRES class.
 * 
 */
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <cstring>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/random/SketchingHandler.hpp>
#include "LinSolverIterativeRandFGMRES.hpp"

namespace ReSolve
{
  using out = io::Logger;

  LinSolverIterativeRandFGMRES::LinSolverIterativeRandFGMRES(MatrixHandler*  matrix_handler,
                                                             VectorHandler*  vector_handler, 
                                                             SketchingMethod rand_method, 
                                                             GramSchmidt*    gs)
  {
    tol_ = 1e-14; //default
    maxit_= 100; //default
    restart_ = 10;
    conv_cond_ = 0;//default
    flexible_ = true;

    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    sketching_method_ = rand_method;
    GS_ = gs;

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
    tol_ = tol; 
    maxit_= maxit; 
    restart_ = restart;
    conv_cond_ = conv_cond;
    flexible_ = true;

    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    sketching_method_ = rand_method;
    GS_ = gs;

    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeRandFGMRES::~LinSolverIterativeRandFGMRES()
  {
    if (is_solver_set_) {
      freeSolverData();
      is_solver_set_ = false;
    }

    if (is_sketching_set_) {
      freeSketchingData();
      is_sketching_set_ = false;
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
    if (n_ != A->getNumRows()) {
      if (is_solver_set_) {
        out::warning() << "Matrix size changed. Reallocating solver ...\n";
        freeSolverData();
        is_solver_set_ = false;
      }

      if (is_sketching_set_) {
        out::warning() << "Matrix size changed. Reallocating solver ...\n";
        freeSketchingData();
        is_sketching_set_ = false;
      }
    }

    A_ = A;
    n_ = A_->getNumRows();

    if (!is_solver_set_) {
      allocateSolverData();
      is_solver_set_ = true;
    }

    if (!is_sketching_set_) {
      allocateSketchingData();
      is_sketching_set_ = true;
    }

    GS_->setup(k_rand_, restart_);
    // GS_->setup(n_, restart_);

    return 0;
  }

  int  LinSolverIterativeRandFGMRES::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    // io::Logger::setVerbosity(io::Logger::EVERYTHING);

    int outer_flag = 1;
    int notconv = 1; 
    index_type i = 0;
    int it = 0;
    int j;
    int k;
    int k1;

    real_type t;
    real_type rnorm;
    real_type bnorm;
    // real_type rnorm_aux;
    real_type tolrel;
    vector_type* vec_v = new vector_type(n_);
    vector_type* vec_z = new vector_type(n_);
    vector_type* vec_s = new vector_type(k_rand_);
    //V[0] = b-A*x_0
    //debug
    vec_Z_->setToZero(memspace_);
    vec_V_->setToZero(memspace_);

    rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);  
    matrix_handler_->matvec(A_, x, vec_V_, &MINUSONE, &ONE, memspace_); 

    vec_v->setData(vec_V_->getVectorData(0, memspace_), memspace_);
    vec_s->setData(vec_S_->getVectorData(0, memspace_), memspace_);

    sketching_handler_->Theta(vec_v, vec_s);

    if (sketching_method_ == fwht) {
      vector_handler_->scal(&one_over_k_, vec_s, memspace_);
    }
    mem_.deviceSynchronize();

    rnorm = 0.0;
    bnorm = vector_handler_->dot(rhs, rhs, memspace_);
    rnorm = vector_handler_->dot(vec_s, vec_s, memspace_);
    rnorm = std::sqrt(rnorm); // rnorm = ||V_1||
    bnorm = std::sqrt(bnorm);
    io::Logger::misc() << "it 0: norm of residual "
                       << std::scientific << std::setprecision(16) 
                       << rnorm << " Norm of rhs: " << bnorm << "\n";
    initial_residual_norm_ = rnorm;
    while(outer_flag) {
      // check if maybe residual is already small enough?
      if (it == 0) {
        tolrel = tol_ * rnorm;
        if (std::abs(tolrel) < 1e-16) {
          tolrel = 1e-16;
        }
      }

      bool exit_cond = false;
      switch (conv_cond_)
      {
        case 0:
          exit_cond = ((std::abs(rnorm - ZERO) <= EPSILON));
          break;
        case 1:
          exit_cond = ((std::abs(rnorm - ZERO) <= EPSILON) || (rnorm < tol_));
          break;
        case 2:
          exit_cond = ((std::abs(rnorm - ZERO) <= EPSILON) || (rnorm < (tol_*bnorm)));
          break;
      }

      if (exit_cond) {
        outer_flag = 0;
        final_residual_norm_ = rnorm;
        initial_residual_norm_ = rnorm;
        total_iters_ = 0;
        break;
      }

      // normalize first vector
      t = 1.0 / rnorm;
      vector_handler_->scal(&t, vec_V_, memspace_);
      vector_handler_->scal(&t, vec_S_, memspace_);

      mem_.deviceSynchronize();
      // initialize norm history
      h_rs_[0] = rnorm;
      i = -1;
      notconv = 1;

      while((notconv) && (it < maxit_)) {
        i++;
        it++;

        // Z_i = (LU)^{-1}*V_i
        vec_v->setData(vec_V_->getVectorData(i, memspace_), memspace_);
        if (flexible_) {
          vec_z->setData(vec_Z_->getVectorData(i, memspace_), memspace_);
        } else {
          vec_z->setData(vec_Z_->getVectorData(0, memspace_), memspace_);
        }
        this->precV(vec_v, vec_z);

        mem_.deviceSynchronize();

        // V_{i+1}=A*Z_i
        vec_v->setData(vec_V_->getVectorData(i + 1, memspace_), memspace_);

        matrix_handler_->matvec(A_, vec_z, vec_v, &ONE, &ZERO, memspace_); 

        // orthogonalize V[i+1], form a column of h_H_
        // this is where it differs from normal solver GS
        vec_s->setData(vec_S_->getVectorData(i + 1, memspace_), memspace_);
        sketching_handler_->Theta(vec_v, vec_s); 
        if (sketching_method_ == fwht) {
          vector_handler_->scal(&one_over_k_, vec_s, memspace_);
        }
        mem_.deviceSynchronize();
        GS_->orthogonalize(k_rand_, vec_S_, h_H_, i);
        // now post-process
        if (memspace_ == memory::DEVICE) {
          mem_.copyArrayHostToDevice(d_aux_, &h_H_[i * (restart_ + 1)], i + 2);
        } else {
          mem_.copyArrayHostToHost(d_aux_, &h_H_[i * (restart_ + 1)], i + 2);
        }
        vec_z->setData(d_aux_, memspace_);
        vec_z->setCurrentSize(i + 1);
        // V(:, i+1) = w - V(:, 1:i)*d_H_col = V(:, i+1) - d_H_col*V(:,1:i); 

        vector_handler_->gemv('N', n_, i + 1, &MINUSONE, &ONE, vec_V_, vec_z, vec_v, memspace_ );  

        vec_z->setCurrentSize(n_);
        t = 1.0 / h_H_[i * (restart_ + 1) + i + 1];
        vector_handler_->scal(&t, vec_v, memspace_);  
        mem_.deviceSynchronize();
        vec_s->setData(vec_S_->getVectorData(i + 1, memspace_), memspace_);

        if (i != 0) {
          for (int k = 1; k <= i; k++) {
            k1 = k - 1;
            t = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k] = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if (i != 0)
        real_type Hii = h_H_[i * (restart_ + 1) + i];
        real_type Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
        real_type gam = std::sqrt(Hii * Hii + Hii1 * Hii1);

        if(std::abs(gam - ZERO) <= EPSILON) {
          gam = EPSMAC;
        }

        /* next Given's rotation */
        h_c_[i] = Hii / gam;
        h_s_[i] = Hii1 / gam;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i] = h_c_[i] * h_rs_[i];

        h_H_[(i) * (restart_ + 1) + (i)]     = h_c_[i] * Hii  + h_s_[i] * Hii1;
        h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

        // residual norm estimate
        rnorm = std::abs(h_rs_[i + 1]);

        io::Logger::misc() << "it: "<< it << " --> norm of the residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
        // check convergence
        if (i + 1 >= restart_ || rnorm <= tolrel || it >= maxit_) {
          notconv = 0;
        }
      } // inner while

      io::Logger::misc() << "End of cycle, ESTIMATED norm of residual "
                         << std::scientific << std::setprecision(16)
                         << rnorm << "\n";
      // solve tri system
      h_rs_[i] = h_rs_[i] / h_H_[i * (restart_ + 1) + i];
      for (int ii = 2; ii <= i + 1; ii++) {
        k = i - ii + 1;
        k1 = k + 1;
        t = h_rs_[k];
        for(j = k1; j <= i; j++) {
          t -= h_H_[j * (restart_ + 1) + k] * h_rs_[j];
        }
        h_rs_[k] = t / h_H_[k * (restart_ + 1) + k];
      }

      // get solution
      if (flexible_) {
        for (j = 0; j <= i; j++) {
          vec_z->setData(vec_Z_->getVectorData(j, memspace_), memspace_);
          vector_handler_->axpy(&h_rs_[j], vec_z, x, memspace_);
        }
      } else {
        vec_Z_->setToZero(0, memspace_);
        vec_z->setData( vec_Z_->getVectorData(0, memspace_), memspace_);
        for(j = 0; j <= i; j++) {
          vec_v->setData(vec_V_->getVectorData(j, memspace_), memspace_);
          vector_handler_->axpy(&h_rs_[j], vec_v, vec_z, memspace_);
        }
        // now multiply d_Z by precon

        vec_v->setData(vec_V_->getData(memspace_), memspace_);
        this->precV(vec_z, vec_v);
        // and add to x 
        vector_handler_->axpy(&ONE, vec_v, x, memspace_);
      }

      /* test solution */
      if(rnorm <= tolrel || it >= maxit_) {
        // rnorm_aux = rnorm;
        outer_flag = 0;
      }

      rhs->copyDataTo(vec_V_->getData(memspace_), 0, memspace_);  
      matrix_handler_->matvec(A_, x, vec_V_, &MINUSONE, &ONE, memspace_); 
      if (outer_flag) {

        sketching_handler_->reset();

        if (sketching_method_ == cs) {
          vec_S_->setToZero(memspace_);
        }
        vec_v->setData(vec_V_->getVectorData(0, memspace_), memspace_);
        vec_s->setData(vec_S_->getVectorData(0, memspace_), memspace_);
        sketching_handler_->Theta(vec_v, vec_s);
        if (sketching_method_ == fwht) {
          vector_handler_->scal(&one_over_k_, vec_s, memspace_);
        }
        mem_.deviceSynchronize();
        rnorm = vector_handler_->dot(vec_S_, vec_S_, memspace_);
        // rnorm = ||S_0||
        rnorm = std::sqrt(rnorm);
      }

      if (!outer_flag) {
        rnorm = vector_handler_->dot(vec_V_, vec_V_, memspace_);
        // rnorm = ||V_0||
        rnorm = std::sqrt(rnorm);

        io::Logger::misc() << "End of cycle, COMPUTED norm of residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";

        final_residual_norm_ = rnorm;
        total_iters_ = it;
      }
    } // outer while
    return 0;
  }

  int  LinSolverIterativeRandFGMRES::setupPreconditioner(std::string type, LinSolverDirect* LU_solver)
  {
    if (type != "LU") {
      out::warning() << "Only cusolverRf tri solve can be used as a preconditioner at this time." << std::endl;
      return 1;
    } else {
      LU_solver_ = LU_solver;  
      return 0;
    }

  }

  index_type  LinSolverIterativeRandFGMRES::getKrand()
  {
    return k_rand_;
  }

  int  LinSolverIterativeRandFGMRES::resetMatrix(matrix::Sparse* new_matrix)
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
    if (is_sketching_set_) {
      if (method == sketching_method_) {
        out::misc() << "Keeping sketching method " << method << "\n";
        return 0;
      }
      out::misc() << "Deleting sketching method " << sketching_method_ << "\n";
      freeSketchingData();
      is_sketching_set_ = false;
    }

    // If solver is set, go ahead and create sketching, otherwise just set sketching method.
    sketching_method_ = method;
    if (is_solver_set_) {
      out::misc() << "Allocating sketching method " << sketching_method_ << "\n";
      allocateSketchingData();
      is_sketching_set_ = true;
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
    if (restart_ == restart) {
      return 0;
    }

    // Set new restart value
    restart_ = restart;

    // If solver is already set, reallocate solver data
    if (is_solver_set_) {
      freeSolverData();
      allocateSolverData();
    }

    // If sketching has been set, reallocate sketching data
    if (is_sketching_set_) {
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
    if (vec_Z_) {
      delete vec_Z_;
      if (is_flexible) {
        vec_Z_ = new vector_type(n_, restart_ + 1);
      } else {
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
  int LinSolverIterativeRandFGMRES::setConvergenceCondition(index_type conv_cond)
  {
    conv_cond_ = conv_cond;
    return 0;
  }

  index_type  LinSolverIterativeRandFGMRES::getRestart() const
  {
    return restart_;
  }

  index_type  LinSolverIterativeRandFGMRES::getConvCond() const
  {
    return conv_cond_;
  }

  bool  LinSolverIterativeRandFGMRES::getFlexible() const
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
    if (flexible_) {
      vec_Z_ = new vector_type(n_, restart_ + 1);
    } else {
      // otherwise Z is just one vector, not multivector and we dont keep it
      vec_Z_ = new vector_type(n_);
    }
    vec_Z_->allocate(memspace_);   
    h_H_  = new real_type[restart_ * (restart_ + 1)];
    h_c_  = new real_type[restart_];      // needed for givens
    h_s_  = new real_type[restart_];      // same
    h_rs_ = new real_type[restart_ + 1];  // for residual norm history
    if (memspace_ == memory::DEVICE) {
      mem_.allocateArrayOnDevice(&d_aux_, restart_ + 1); 
    } else {
      d_aux_  = new real_type[restart_ + 1];
    }
    return 0;
  }

  int LinSolverIterativeRandFGMRES::freeSolverData()
  {
    delete [] h_H_ ;
    delete [] h_c_ ;
    delete [] h_s_ ;
    delete [] h_rs_;
    if (memspace_ == memory::DEVICE) { 
      mem_.deleteOnDevice(d_aux_);
    } else {
      delete [] d_aux_;
    }
    delete vec_V_;   
    delete vec_Z_;

    h_H_   = nullptr;
    h_c_   = nullptr;
    h_s_   = nullptr;
    h_rs_  = nullptr;
    d_aux_ = nullptr;
    vec_V_   = nullptr;
    vec_Z_   = nullptr;

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
    switch (sketching_method_) {
      case cs:
        if (std::ceil(restart_ * std::log(n_)) < k_rand_) {
          k_rand_ = static_cast<index_type>(std::ceil(restart_ * std::log(static_cast<real_type>(n_))));
        }
        sketching_handler_ = new SketchingHandler(sketching_method_, device_type_);
        // set k and n 
        break;
      case fwht:
        if (std::ceil(2.0 * restart_ * std::log(n_) / std::log(restart_)) < k_rand_) {
          k_rand_ = static_cast<index_type>(std::ceil(2.0 * restart_ * std::log(n_) / std::log(restart_)));
        }
        sketching_handler_ = new SketchingHandler(sketching_method_, device_type_);
        break;
      default:
        io::Logger::warning() << "Wrong sketching method, setting to default (CountSketch)\n"; 
        sketching_method_ = cs;
        if (std::ceil(restart_ * std::log(n_)) < k_rand_) {
          k_rand_ = static_cast<index_type>(std::ceil(restart_ * std::log(n_)));
        }
        sketching_handler_ = new SketchingHandler(cs, device_type_);
        break;
    }

    one_over_k_ = 1.0 / std::sqrt((real_type) k_rand_);
    vec_S_ = new vector_type(k_rand_, restart_ + 1);
    vec_S_->allocate(memspace_);      
    if (sketching_method_ == cs) {
      vec_S_->setToZero(memspace_);
    }

    sketching_handler_->setup(n_, k_rand_);
    return 0;
  }

  int LinSolverIterativeRandFGMRES::freeSketchingData()
  {
    delete vec_S_;
    delete sketching_handler_;

    vec_S_ = nullptr;
    sketching_handler_ = nullptr;

    return 0;
  }

  void LinSolverIterativeRandFGMRES::precV(vector_type* rhs, vector_type* x)
  { 
    LU_solver_->solve(rhs, x);
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

    if ((is_matrix_handler_cuda != is_vector_handler_cuda) || 
        (is_matrix_handler_hip  != is_vector_handler_hip )) {
      out::error() << "Matrix and vector handler backends are incompatible!\n";  
    }

    if (is_matrix_handler_cuda) {
      memspace_ = memory::DEVICE;
      device_type_ = memory::CUDADEVICE;
    } else if (is_matrix_handler_hip) {
      memspace_ = memory::DEVICE;
      device_type_ = memory::HIPDEVICE;
    } else {
      memspace_ = memory::HOST;
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
