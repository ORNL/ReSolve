#include <iostream>
#include <cassert>
#include <cmath>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include "LinSolverIterativeFGMRES.hpp"

namespace ReSolve
{
  using out = io::Logger;

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES()
  {
    this->matrix_handler_ = nullptr;
    this->vector_handler_ = nullptr;
    tol_ = 1e-14; //default
    maxit_= 100; //default
    restart_ = 10;
    conv_cond_ = 0;//default

    d_V_ = nullptr;
    d_Z_ = nullptr;
  }

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES(MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
  {
    this->matrix_handler_ = matrix_handler;
    this->vector_handler_ = vector_handler;
    this->GS_ = gs;

    tol_ = 1e-14; //default
    maxit_= 100; //default
    restart_ = 10;
    conv_cond_ = 0;//default

    d_V_ = nullptr;
    d_Z_ = nullptr;
  }

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES(index_type restart, 
                                                     real_type  tol,
                                                     index_type maxit,
                                                     index_type conv_cond,
                                                     MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
  {
    this->matrix_handler_ = matrix_handler;
    this->vector_handler_ = vector_handler;
    this->GS_ = gs;

    tol_ = tol; 
    maxit_= maxit; 
    restart_ = restart;
    conv_cond_ = conv_cond;

    d_V_ = nullptr;
    d_Z_ = nullptr;

  }

  LinSolverIterativeFGMRES::~LinSolverIterativeFGMRES()
  {
    if (d_V_ != nullptr) {
      // cudaFree(d_V_);
      delete d_V_;   
    }

    if (d_Z_ != nullptr) {
      //      cudaFree(d_Z_);
      delete d_Z_;   
    }

  }

  int LinSolverIterativeFGMRES::setup(matrix::Sparse* A)
  {
    this->A_ = A;
    n_ = A_->getNumRows();

    d_V_ = new vector_type(n_, restart_ + 1);
    d_V_->allocate("cuda");      
    d_Z_ = new vector_type(n_, restart_ + 1);
    d_Z_->allocate("cuda");      
    h_H_  = new real_type[restart_ * (restart_ + 1)];
    h_c_  = new real_type[restart_];      // needed for givens
    h_s_  = new real_type[restart_];      // same
    h_rs_ = new real_type[restart_ + 1]; // for residual norm history

    return 0;
  }

  int  LinSolverIterativeFGMRES::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    int outer_flag = 1;
    int notconv = 1; 
    int i = 0;
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
    //V[0] = b-A*x_0

    rhs->deepCopyVectorData(d_V_->getData("cuda"), 0, "cuda");  
    matrix_handler_->matvec(A_, x, d_V_, &MINUSONE, &ONE, "csr", "cuda"); 
    rnorm = 0.0;
    bnorm = vector_handler_->dot(rhs, rhs, "cuda");
    rnorm = vector_handler_->dot(d_V_, d_V_, "cuda");

    //rnorm = ||V_1||
    rnorm = sqrt(rnorm);
    bnorm = sqrt(bnorm);
    initial_residual_norm_ = rnorm;
    while(outer_flag) {
      // check if maybe residual is already small enough?
      if(it == 0) {
        tolrel = tol_ * rnorm;
        if(fabs(tolrel) < 1e-16) {
          tolrel = 1e-16;
        }
      }
      int exit_cond = 0;
      if (conv_cond_ == 0){
        exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON));
      } else {
        if (conv_cond_ == 1){
          exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < tol_));
        } else {
          if (conv_cond_ == 2){
            exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < (tol_*bnorm)));
          }
        }
      }
      if (exit_cond) {
        outer_flag = 0;
        final_residual_norm_ = rnorm;
        initial_residual_norm_ = rnorm;
        fgmres_iters_ = 0;
        break;
      }

      // normalize first vector
      t = 1.0 / rnorm;
      vector_handler_->scal(&t, d_V_, "cuda");
      // initialize norm history
      h_rs_[0] = rnorm;
      i = -1;
      notconv = 1;

      while((notconv) && (it < maxit_)) {
        i++;
        it++;

        // Z_i = (LU)^{-1}*V_i

        vec_v->setData( d_V_->getVectorData(i, "cuda"), "cuda");
        vec_z->setData( d_Z_->getVectorData(i, "cuda"), "cuda");
        this->precV(vec_v, vec_z);
        mem_.deviceSynchronize();

        // V_{i+1}=A*Z_i

        vec_v->setData( d_V_->getVectorData(i + 1, "cuda"), "cuda");

        matrix_handler_->matvec(A_, vec_z, vec_v, &ONE, &ZERO,"csr", "cuda"); 

        // orthogonalize V[i+1], form a column of h_H_

        GS_->orthogonalize(n_, d_V_, h_H_, i, "cuda");  ;
        if(i != 0) {
          for(int k = 1; k <= i; k++) {
            k1 = k - 1;
            t = h_H_[i * (restart_ + 1) + k1];
            h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
            h_H_[i * (restart_ + 1) + k] = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
          }
        } // if i!=0

        double Hii = h_H_[i * (restart_ + 1) + i];
        double Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
        double gam = sqrt(Hii * Hii + Hii1 * Hii1);

        if(fabs(gam - ZERO) <= EPSILON) {
          gam = EPSMAC;
        }

        /* next Given's rotation */
        h_c_[i] = Hii / gam;
        h_s_[i] = Hii1 / gam;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i] = h_c_[i] * h_rs_[i];

        h_H_[(i) * (restart_ + 1) + (i)] = h_c_[i] * Hii + h_s_[i] * Hii1;
        h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

        // residual norm estimate
        rnorm = fabs(h_rs_[i + 1]);
        // check convergence
        if(i + 1 >= restart_ || rnorm <= tolrel || it >= maxit_) {
          notconv = 0;
        }
      } // inner while

      // solve tri system
      h_rs_[i] = h_rs_[i] / h_H_[i * (restart_ + 1) + i];
      for(int ii = 2; ii <= i + 1; ii++) {
        k = i - ii + 1;
        k1 = k + 1;
        t = h_rs_[k];
        for(j = k1; j <= i; j++) {
          t -= h_H_[j * (restart_ + 1) + k] * h_rs_[j];
        }
        h_rs_[k] = t / h_H_[k * (restart_ + 1) + k];
      }

      // get solution
      for(j = 0; j <= i; j++) {
        vec_z->setData( d_Z_->getVectorData(j, "cuda"), "cuda");
        vector_handler_->axpy(&h_rs_[j], vec_z, x, "cuda");
      }

      /* test solution */

      if(rnorm <= tolrel || it >= maxit_) {
        // rnorm_aux = rnorm;
        outer_flag = 0;
      }

      rhs->deepCopyVectorData(d_V_->getData("cuda"), 0, "cuda");  
      matrix_handler_->matvec(A_, x, d_V_, &MINUSONE, &ONE,"csr", "cuda"); 
      rnorm = vector_handler_->dot(d_V_, d_V_, "cuda");
      // rnorm = ||V_1||
      rnorm = sqrt(rnorm);

      if(!outer_flag) {
        final_residual_norm_ = rnorm;
        fgmres_iters_ = it;
      }
    } // outer while
    return 0;
  }

  int  LinSolverIterativeFGMRES::setupPreconditioner(std::string name, LinSolverDirect* LU_solver)
  {
    if (name != "CuSolverRf") {
      out::warning() << "Only cusolverRf tri solve can be used as a preconditioner at this time." << std::endl;
      return 1;
    } else {
      LU_solver_ = LU_solver;  
      return 0;
    }

  }

  real_type  LinSolverIterativeFGMRES::getTol()
  {
    return tol_;
  }

  index_type  LinSolverIterativeFGMRES::getMaxit()
  {
    return maxit_;
  }

  index_type  LinSolverIterativeFGMRES::getRestart()
  {
    return restart_;
  }

  index_type  LinSolverIterativeFGMRES::getConvCond()
  {
    return conv_cond_;
  }

  void  LinSolverIterativeFGMRES::setTol(real_type new_tol)
  {
    this->tol_ = new_tol;
  }

  void  LinSolverIterativeFGMRES::setMaxit(index_type new_maxit)
  {
    this->maxit_ = new_maxit;
  }

  void  LinSolverIterativeFGMRES::setRestart(index_type new_restart)
  {
    this->restart_ = new_restart;
  }

  void  LinSolverIterativeFGMRES::setConvCond(index_type new_conv_cond)
  {
    this->conv_cond_ = new_conv_cond;
  }

  int  LinSolverIterativeFGMRES::resetMatrix(matrix::Sparse* new_matrix)
  {
    A_ = new_matrix;
    matrix_handler_->setValuesChanged(true, "cuda");
    return 0;
  }



  void  LinSolverIterativeFGMRES::precV(vector_type* rhs, vector_type* x)
  { 
    LU_solver_->solve(rhs, x);
    //  x->update(rhs->getData("cuda"), "cuda", "cuda");
  }

  real_type LinSolverIterativeFGMRES::getFinalResidualNorm()
  {
    return final_residual_norm_;
  }

  real_type LinSolverIterativeFGMRES::getInitResidualNorm()
  {
    return initial_residual_norm_;
  }

  index_type LinSolverIterativeFGMRES::getNumIter()
  {
    return fgmres_iters_;
  }
}//namespace
