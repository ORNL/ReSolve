#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include "LinSolverIterativeRandFGMRES.hpp"
#include <resolve/RandSketchingCountSketch.hpp> 
#include <resolve/RandSketchingFWHT.hpp> 

namespace ReSolve
{
  using out = io::Logger;
 
  LinSolverIterativeRandFGMRES::LinSolverIterativeRandFGMRES(MatrixHandler* matrix_handler,
                                                             VectorHandler* vector_handler, 
                                                             SketchingMethod rand_method, 
                                                             GramSchmidt*   gs)
  {
    this->matrix_handler_ = matrix_handler;
    this->vector_handler_ = vector_handler;
    this->rand_method_ = rand_method;
    this->GS_ = gs;

    setMemorySpace();

    tol_ = 1e-14; //default
    maxit_= 100; //default
    restart_ = 10;
    conv_cond_ = 0;//default
    flexible_ = true;

    d_V_ = nullptr;
    d_Z_ = nullptr;
  }

  LinSolverIterativeRandFGMRES::LinSolverIterativeRandFGMRES(index_type restart, 
                                                             real_type  tol,
                                                             index_type maxit,
                                                             index_type conv_cond,
                                                             MatrixHandler* matrix_handler,
                                                             VectorHandler* vector_handler,
                                                             SketchingMethod rand_method, 
                                                             GramSchmidt*   gs)
  {
    this->matrix_handler_ = matrix_handler;
    this->vector_handler_ = vector_handler;
    this->rand_method_ = rand_method;
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

  LinSolverIterativeRandFGMRES::~LinSolverIterativeRandFGMRES()
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

  int LinSolverIterativeRandFGMRES::setup(matrix::Sparse* A)
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

    mem_.allocateArrayOnDevice(&d_aux_, restart_ + 1); 
    // rand method
    k_rand_ = n_;
    switch(rand_method_) {
      case cs:
        if(ceil(restart_ * log(n_)) < k_rand_) {
          k_rand_ = static_cast<index_type>(std::ceil(restart_ * std::log(static_cast<real_type>(n_))));
        }
        rand_manager_ = new RandSketchingCountSketch();
        //set k and n 
        break;
      case fwht:
        if ( ceil(2.0 * restart_ * log(n_) / log(restart_)) < k_rand_) {
          k_rand_ = static_cast<index_type>(std::ceil(2.0 * restart_ * std::log(n_) / std::log(restart_)));
        }
        rand_manager_ = new RandSketchingFWHT();
        break;
      default:
        io::Logger::warning() << "Wrong sketching method, setting to default (CountSketch)\n"; 
        rand_method_ = cs;
        if(ceil(restart_ * log(n_)) < k_rand_) {
          k_rand_ = static_cast<index_type>(std::ceil(restart_ * std::log(n_)));
        }
        rand_manager_ = new RandSketchingCountSketch();
        break;
    }

    rand_manager_->setup(n_, k_rand_); 

    one_over_k_ = 1.0 / sqrt((real_type) k_rand_);

    d_S_ = new vector_type(k_rand_, restart_ + 1);
    d_S_->allocate(memspace_);      
    if (rand_method_ == cs) {
      d_S_->setToZero(memspace_);
    }
    return 0;
  }

  int  LinSolverIterativeRandFGMRES::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;

    //io::Logger::setVerbosity(io::Logger::EVERYTHING);
    
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
    vector_type* vec_s = new vector_type(k_rand_);
    //V[0] = b-A*x_0
    //debug
    d_Z_->setToZero(memspace_);
    d_V_->setToZero(memspace_);

    rhs->deepCopyVectorData(d_V_->getData(memspace_), 0, memspace_);  
    matrix_handler_->matvec(A_, x, d_V_, &MINUSONE, &ONE, "csr", memspace_); 

    vec_v->setData( d_V_->getVectorData(0, memspace_), memspace_);
    vec_s->setData( d_S_->getVectorData(0, memspace_), memspace_);

    rand_manager_->Theta(vec_v, vec_s);

    if (rand_method_ == fwht){
      //  cublasDscal(cublas_handle, k_rand, &oneOverK, d_S, 1); 
      vector_handler_->scal(&one_over_k_, vec_s, memspace_);
    }
    mem_.deviceSynchronize();

    rnorm = 0.0;
    bnorm = vector_handler_->dot(rhs, rhs, memspace_);
    rnorm = vector_handler_->dot(vec_s, vec_s, memspace_);
    // double rnorm_true = vector_handler_->dot(vec_v, vec_v, memspace_);
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
      if (conv_cond_ == 0) {
        exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON));
      } else if (conv_cond_ == 1) {
        exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < tol_));
      } else if (conv_cond_ == 2) {
        exit_cond =  ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < (tol_*bnorm)));
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
      vector_handler_->scal(&t, d_S_, memspace_);

      mem_.deviceSynchronize();
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
        this->precV(vec_v, vec_z);

        mem_.deviceSynchronize();

        // V_{i+1}=A*Z_i

        vec_v->setData( d_V_->getVectorData(i + 1, memspace_), memspace_);

        matrix_handler_->matvec(A_, vec_z, vec_v, &ONE, &ZERO,"csr", memspace_); 

        // orthogonalize V[i+1], form a column of h_H_
        // this is where it differs from normal solver GS
        vec_s->setData( d_S_->getVectorData(i + 1, memspace_), memspace_);
        rand_manager_->Theta(vec_v, vec_s); 
        if (rand_method_ == fwht){
          //  cublasDscal(cublas_handle, k_rand, &oneOverK, d_S, 1); 
          vector_handler_->scal(&one_over_k_, vec_s, memspace_);
        }
        mem_.deviceSynchronize();
        GS_->orthogonalize(k_rand_, d_S_, h_H_, i); //, memspace_);  
        // now post-process
        //checkCudaErrors(cudaMemcpy(d_Hcolumn, &h_H[i * (restart + 1)], sizeof(double) * (i + 1), cudaMemcpyHostToDevice));
        mem_.copyArrayHostToDevice(d_aux_, &h_H_[i * (restart_ + 1)], i + 2);
        vec_z->setData(d_aux_, memspace_);
        vec_z->setCurrentSize(i + 1);
        //V(:, i+1) =w-V(:, 1:i)*d_H_col = V(:, i+1)-d_H_col * V(:,1:i); 
        //checkCudaErrors( cublasDgemv(cublas_handle, CUBLAS_OP_N, n, i + 1, &minusone, d_V, n, d_Hcolumn, 1,&one , &d_V[n * (i + 1)], 1));

        vector_handler_->gemv('N', n_, i + 1, &MINUSONE, &ONE, d_V_, vec_z, vec_v, memspace_ );  

        vec_z->setCurrentSize(n_);
        t = 1.0 / h_H_[i * (restart_ + 1) + i + 1];
        vector_handler_->scal(&t, vec_v, memspace_);  
        mem_.deviceSynchronize();
        vec_s->setData( d_S_->getVectorData(i + 1, memspace_), memspace_);

        if (i != 0) {
          for (int k = 1; k <= i; k++) {
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
      if (outer_flag) {

        rand_manager_->reset();

        if (rand_method_ == cs) {
          mem_.setZeroArrayOnDevice(d_S_->getData(memspace_), d_S_->getSize() * d_S_->getNumVectors());
        }
        vec_v->setData( d_V_->getVectorData(0, memspace_), memspace_);
        vec_s->setData( d_S_->getVectorData(0, memspace_), memspace_);
        rand_manager_->Theta(vec_v, vec_s);
        if (rand_method_ == fwht){
          //  cublasDscal(cublas_handle, k_rand, &oneOverK, d_S, 1); 
          vector_handler_->scal(&one_over_k_, vec_s, memspace_);
        }
        mem_.deviceSynchronize();
        rnorm = vector_handler_->dot(d_S_, d_S_, memspace_);
        // rnorm = ||S_0||
        rnorm = sqrt(rnorm);
      }

      if (!outer_flag) {
        rnorm = vector_handler_->dot(d_V_, d_V_, memspace_);
        // rnorm = ||V_0||
        rnorm = sqrt(rnorm);
        
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

  //
  // Private methods
  //

  void  LinSolverIterativeRandFGMRES::precV(vector_type* rhs, vector_type* x)
  { 
    LU_solver_->solve(rhs, x);
  }


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

    if (is_matrix_handler_cuda || is_matrix_handler_hip) {
      memspace_ = memory::DEVICE;
    } else {
      memspace_ = memory::HOST;
    }
  }

} // namespace ReSolve
