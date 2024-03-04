#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include "LinSolverIterativeFGMRES.hpp"

#include <chrono>
using namespace std::chrono;
namespace ReSolve
{
  using out = io::Logger;

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES(MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
  {
    this->matrix_handler_ = matrix_handler;
    this->vector_handler_ = vector_handler;
    this->GS_ = gs;

    setMemorySpace();

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

    setMemorySpace();

    tol_ = tol; 
    maxit_= maxit; 
    restart_ = restart;
    conv_cond_ = conv_cond;
    flexible_ = true;

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
    d_V_->allocate(memspace_);      
    if (flexible_) {
      d_Z_ = new vector_type(n_, restart_ + 1);
    } else {
      // otherwise Z is just a one vector, not multivector and we dont keep it
      d_Z_ = new vector_type(n_);
    }
    d_Z_->allocate(memspace_);      
    h_H_  = new real_type[restart_ * (restart_ + 1)];
    h_c_  = new real_type[restart_];      // needed for givens
    h_s_  = new real_type[restart_];      // same
    h_rs_ = new real_type[restart_ + 1]; // for residual norm history

    return 0;
  }

  int  LinSolverIterativeFGMRES::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    io::Logger::setVerbosity(io::Logger::EVERYTHING);
    
    std::chrono::high_resolution_clock::time_point start, stop;
    long long int total_ms = 0; 
    long long int total_gs = 0; 
    
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
    //debug
    d_Z_->setToZero(memspace_);
    d_V_->setToZero(memspace_);

    rhs->deepCopyVectorData(d_V_->getData(memspace_), 0, memspace_);  
    matrix_handler_->matvec(A_, x, d_V_, &MINUSONE, &ONE, "csr", memspace_); 
    rnorm = 0.0;
    bnorm = vector_handler_->dot(rhs, rhs, memspace_);
    rnorm = vector_handler_->dot(d_V_, d_V_, memspace_);
    //rnorm = ||V_1||
    rnorm = sqrt(rnorm);
    bnorm = sqrt(bnorm);
    io::Logger::misc() << "it 0: norm of residual "
                       << std::scientific << std::setprecision(16) 
                       << rnorm << " Norm of rhs: " << bnorm << "\n";
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
        total_iters_ = 0;
        break;
      }

      // normalize first vector
      t = 1.0 / rnorm;
      vector_handler_->scal(&t, d_V_, memspace_);
      // initialize norm history
      h_rs_[0] = rnorm;
      i = -1;
      notconv = 1;

      while((notconv) && (it < maxit_)) {
        i++;
        it++;

        // Z_i = (LU)^{-1}*V_i
        vec_v->setData( d_V_->getVectorData(i, memspace_), memspace_);
        if (flexible_) {
          vec_z->setData( d_Z_->getVectorData(i, memspace_), memspace_);
        } else {
          vec_z->setData( d_Z_->getVectorData(0, memspace_), memspace_);
        }
        auto start = high_resolution_clock::now(); 
        this->precV(vec_v, vec_z);
        mem_.deviceSynchronize();
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        total_ms += (int)  duration.count();

        // V_{i+1}=A*Z_i

        vec_v->setData( d_V_->getVectorData(i + 1, memspace_), memspace_);

        matrix_handler_->matvec(A_, vec_z, vec_v, &ONE, &ZERO,"csr", memspace_); 

        // orthogonalize V[i+1], form a column of h_H_

        start = high_resolution_clock::now();
        GS_->orthogonalize(n_, d_V_, h_H_, i);

        mem_.deviceSynchronize();
        stop = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(stop - start);
        total_gs += (int)  duration1.count();
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
        io::Logger::misc() << "it: "<<it<< " --> norm of the residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
        // check convergence
        if(i + 1 >= restart_ || rnorm <= tolrel || it >= maxit_) {
          notconv = 0;
        }
      } // inner while

      io::Logger::misc() << "End of cycle, ESTIMATED norm of residual "
                         << std::scientific << std::setprecision(16)
                         << rnorm << "\n";
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
      if (flexible_) {
        for(j = 0; j <= i; j++) {
          vec_z->setData( d_Z_->getVectorData(j, memspace_), memspace_);
          vector_handler_->axpy(&h_rs_[j], vec_z, x, memspace_);
        }
      } else {
        mem_.setZeroArrayOnDevice(d_Z_->getData(memspace_), d_Z_->getSize());
        vec_z->setData( d_Z_->getVectorData(0, memspace_), memspace_);
        for(j = 0; j <= i; j++) {
          vec_v->setData( d_V_->getVectorData(j, memspace_), memspace_);
          vector_handler_->axpy(&h_rs_[j], vec_v, vec_z, memspace_);
        }
        // now multiply d_Z by precon

        vec_v->setData( d_V_->getData(memspace_), memspace_);
        this->precV(vec_z, vec_v);
        // and add to x 
        vector_handler_->axpy(&ONE, vec_v, x, memspace_);
      }

      /* test solution */

      if(rnorm <= tolrel || it >= maxit_) {
        // rnorm_aux = rnorm;
        outer_flag = 0;
      }

      rhs->deepCopyVectorData(d_V_->getData(memspace_), 0, memspace_);  
      matrix_handler_->matvec(A_, x, d_V_, &MINUSONE, &ONE,"csr", memspace_); 
      rnorm = vector_handler_->dot(d_V_, d_V_, memspace_);
      // rnorm = ||V_1||
      rnorm = sqrt(rnorm);

      if(!outer_flag) {
        final_residual_norm_ = rnorm;
        total_iters_ = it;
        io::Logger::misc() << "End of cycle, COMPUTED norm of residual "
                           << std::scientific << std::setprecision(16)
                           << rnorm << "\n";
        printf("Prec time: %lld  gs time %lld\n", total_ms, total_gs);
      }
    } // outer while
    return 0;
  }

  int  LinSolverIterativeFGMRES::setupPreconditioner(std::string type, LinSolverDirect* LU_solver)
  {
    if (type != "LU") {
      out::warning() << "Only LU-type solve can be used as a preconditioner at this time." << std::endl;
      return 1;
    } else {
      LU_solver_ = LU_solver;  
      return 0;
    }

  }

  int  LinSolverIterativeFGMRES::resetMatrix(matrix::Sparse* new_matrix)
  {
    A_ = new_matrix;
    matrix_handler_->setValuesChanged(true, memspace_);
    return 0;
  }


  //
  // Private methods
  //

  void  LinSolverIterativeFGMRES::precV(vector_type* rhs, vector_type* x)
  { 
    LU_solver_->solve(rhs, x);
  }

  void LinSolverIterativeFGMRES::setMemorySpace()
  {
    bool is_matrix_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_matrix_handler_hip  = matrix_handler_->getIsHipEnabled();
    bool is_vector_handler_cuda = matrix_handler_->getIsCudaEnabled();
    bool is_vector_handler_hip  = matrix_handler_->getIsHipEnabled();

    if ((is_matrix_handler_cuda != is_vector_handler_cuda) || 
        (is_matrix_handler_hip  != is_vector_handler_hip )) {
      out::error() << "Matrix and vector handler backends are incompatible!\n";  
    }

    if (is_matrix_handler_cuda || is_matrix_handler_hip) {
      memspace_ = memory::DEVICE;
    } else {
      memspace_ = memory::HOST;
    }
  }

}//namespace
