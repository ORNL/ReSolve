#include "LinSolverIterativeFGMRES.hpp"
#include <iostream>
#include <cassert>

namespace ReSolve 
{

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES()
  {
    this->matrix_handler_ = nullptr;
    this->vector_handler_ = nullptr;
    tol_ = 1e-14; //default
    maxit_= 100; //default
    restart_ = 10;
    orth_option_ = "cgs2"; //MGS, slow
    conv_cond_ = 0;//default

    d_V_ = nullptr;
    d_Z_ = nullptr;
  }

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES(MatrixHandler* matrix_handler, VectorHandler* vector_handler)
  {
    this->matrix_handler_ = matrix_handler;
    this->vector_handler_ = vector_handler;

    tol_ = 1e-14; //default
    maxit_= 100; //default
    restart_ = 10;
    orth_option_ = "cgs2"; //MGS, slow
    conv_cond_ = 0;//default

    d_V_ = nullptr;
    d_Z_ = nullptr;
  }

  LinSolverIterativeFGMRES::LinSolverIterativeFGMRES(index_type restart, real_type tol, index_type maxit, std::string GS_version, index_type conv_cond, MatrixHandler* matrix_handler, VectorHandler* vector_handler)
  {
    this->matrix_handler_ = matrix_handler;
    this->vector_handler_ = vector_handler;

    tol_ = tol; 
    maxit_= maxit; 
    restart_ = restart;
    orth_option_ = GS_version;
    conv_cond_ = conv_cond;

    d_V_ = nullptr;
    d_Z_ = nullptr;

  }

  LinSolverIterativeFGMRES::~LinSolverIterativeFGMRES()
  {
    if (d_V_ != nullptr) {
      cudaFree(d_V_);
    }

    if (d_Z_ != nullptr) {
      cudaFree(d_Z_);
    }

    if (d_Hcolumn_ != nullptr) {
      cudaFree(d_Hcolumn_);
    }
    if (d_rvGPU_ != nullptr) {
      cudaFree(d_rvGPU_);
    }
  }

  int LinSolverIterativeFGMRES::setup(matrix::Sparse* A)
  {
    this->A_ = A;
    n_ = A_->getNumRows();
    cudaMalloc(&(d_V_),      n_ * (restart_ + 1) * sizeof(real_type));
    cudaMalloc(&(d_Z_),      n_ * (restart_ + 1) * sizeof(real_type));
    cudaMalloc(&(d_rvGPU_),   2 * (restart_ + 1) * sizeof(real_type));
    cudaMalloc(&(d_Hcolumn_), 2 * (restart_ + 1) * (restart_ + 1) * sizeof(real_type));

    h_H_  = new real_type[restart_ * (restart_ + 1)];
    h_c_  = new real_type[restart_];      // needed for givens
    h_s_  = new real_type[restart_];      // same
    h_rs_ = new real_type[restart_ + 1]; // for residual norm history

    // for specific orthogonalization options, need a little more memory
    if(orth_option_ == "mgs_two_synch" || orth_option_ == "mgs_pm") {
      h_L_  = new real_type[restart_ * (restart_ + 1)];
      h_rv_ = new real_type[restart_ + 1];
    }

    if(orth_option_ == "cgs2") {
      h_aux_ = new real_type[restart_ + 1];
      cudaMalloc(&(d_H_col_), (restart_ + 1) * sizeof(real_type));
    }

    if(orth_option_ == "mgs_pm") {
      h_aux_ = new real_type[restart_ + 1];
    }
    return 0;
  }

  int  LinSolverIterativeFGMRES::solve(vector_type* rhs, vector_type* x)
  {
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
    cudaMemcpy(&(d_V_[0]), rhs->getData("cuda"), sizeof(real_type) * n_, cudaMemcpyDeviceToDevice);
    //cudaMatvec(d_x, d_V_, "residual");
    vec_v->setData(d_V_, "cuda");

    matrix_handler_->matvec(A_, x, vec_v, &minusone_, &one_,"csr", "cuda"); 
    rnorm = 0.0;
    //cublasDdot (cublas_handle_,  n_, d_b, 1, d_b, 1, &bnorm);
    bnorm = vector_handler_->dot(rhs, rhs, "cuda");
    //cublasDdot (cublas_handle_,  n_, d_V_, 1, d_V_, 1, &rnorm);
    rnorm = vector_handler_->dot(vec_v, vec_v, "cuda");
    //rnorm = ||V_1||
    rnorm = sqrt(rnorm);
    bnorm = sqrt(bnorm);
    //printf("FGMRES: init rel norm of R %16.16e \n", rnorm/bnorm);
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
      //      cublasDscal(cublas_handle_, n_, &t, d_V_, 1);
      vector_handler_->scal(&t, vec_v, "cuda");
      // initialize norm history
      h_rs_[0] = rnorm;
      i = -1;
      notconv = 1;

      while((notconv) && (it < maxit_)) {
        i++;
        it++;
        // Z_i = (LU)^{-1}*V_i

        //    cudaMemcpy(&d_Z_[i * n_], &d_V_[i * n_], sizeof(double) * n_, cudaMemcpyDeviceToDevice);
        vec_v->setData(&d_V_[i * n_], "cuda");
        vec_z->setData(&d_Z_[i * n_], "cuda");
        this->precV(vec_v, vec_z);
        //        checkCudaErrors(cusolverRfSolve(cusolverrf_handle_, d_P_, d_Q_, 1, d_T_, n_, &d_Z_[i * n_], n_));
        cudaDeviceSynchronize();
        // V_{i+1}=A*Z_i

        vec_v->setData(&d_V_[(i + 1) * n_], "cuda");

        matrix_handler_->matvec(A_, vec_z, vec_v, &one_, &zero_,"csr", "cuda"); 
        //   cudaMatvec(&d_Z_[i * n_], &d_V_[(i + 1) * n_], "matvec");
        // orthogonalize V[i+1], form a column of h_L
        GramSchmidt(i);

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
      // vec_v->setData(x, "cuda");
      for(j = 0; j <= i; j++) {
        vec_z->setData(&d_Z_[j * n_], "cuda");

        //cublasDaxpy(cublas_handle_, n_, &h_rs_[j], &d_Z_[j * n_], 1, d_x, 1);
        vector_handler_->axpy(&h_rs_[j], vec_z, x, "cuda");
      }

      /* test solution */

      if(rnorm <= tolrel || it >= maxit_) {
        // rnorm_aux = rnorm;
        outer_flag = 0;
      }

      cudaMemcpy(&d_V_[0], rhs->getData("cuda"), sizeof(double)*n_, cudaMemcpyDeviceToDevice);
      //cudaMatvec(d_x, d_V_, "residual");
      vec_v->setData(d_V_, "cuda");
      matrix_handler_->matvec(A_, x, vec_v, &minusone_, &one_,"csr", "cuda"); 
      rnorm = vector_handler_->dot(vec_v, vec_v, "cuda");
      //cublasDdot(cublas_handle_, n_, d_V_, 1, d_V_, 1, &rnorm);
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
      std::cout<<"Only cusolverRf tri solve can be used as a preconditioner at thistime"<<std::endl;
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

  void  LinSolverIterativeFGMRES::setGSversion(std::string new_GS)
  {
    this->orth_option_ =  new_GS;
  }

  void  LinSolverIterativeFGMRES::setConvCond(index_type new_conv_cond)
  {
    this->conv_cond_ = new_conv_cond;
  }

  int  LinSolverIterativeFGMRES::resetMatrix(matrix::Sparse* new_matrix)
  {
    A_ = new_matrix;
    matrix_handler_->setValuesChanged(true);
    return 0;
  }

  int  LinSolverIterativeFGMRES::GramSchmidt(index_type i) 
  {
    double t;
    double s;
    int sw = 0;
    if(orth_option_ == "mgs") {
      sw = 0;
    } else {
      if(orth_option_ == "cgs2") {
        sw = 1;
      } else {
        if(orth_option_ == "mgs_two_synch") {
          sw = 2;
        } else {
          if(orth_option_ == "mgs_pm") {
            sw = 3;
          } else {
            // display error message and set sw = 0;
            /*
               nlp_->log->printf(hovWarning, 
               "Wrong Gram-Schmidt option. Setting default (modified Gram-Schmidt, mgs) ...\n");
               */
            sw = 0;
          }
        }
      }
    }

    vector_type* vec_w = new vector_type(n_);
    vector_type* vec_v = new vector_type(n_);
    switch (sw){
      case 0: //mgs

        vec_w->setData( &d_V_[(i + 1) * n_], "cuda");
        for(int j=0; j<=i; ++j) {
          t=0.0;
          vec_v->setData( &d_V_[j * n_], "cuda");
          t = vector_handler_->dot(vec_v, vec_w, "cuda");  
          //cublasDdot (cublas_handle_,  n_, &d_V_[j*n_], 1, &d_V_[(i+1)*n_], 1, &t);

          h_H_[ i * (restart_ + 1) + j] = t; 
          t *= -1.0;

          /*					cublasDaxpy(cublas_handle_,
                      n_,
                      &t,
                      &d_V_[j*n_],
                      1,
                      &d_V_[(i+1)*n_],
                      1);
                      */

          vector_handler_->axpy(&t, vec_v, vec_w, "cuda");  
        }
        t = 0.0;
        //				cublasDdot(cublas_handle_,  n_, &d_V_[(i+1)*n_], 1, &d_V_[(i+1)*n_], 1, &t);

        t = vector_handler_->dot(vec_w, vec_w, "cuda");  
        //set the last entry in Hessenberg matrix
        t = sqrt(t);
        h_H_[(i) * (restart_ + 1) + i + 1] = t;    
        if(t != 0.0) {
          t = 1.0/t;
          //cublasDscal(cublas_handle_,n_,&t,&d_V_[(i+1)*n_], 1); 

          vector_handler_->scal(&t, vec_w, "cuda");  
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
          return -1;
        }
        
        return 0;
        break;

      case 1://cgs2
        // Hcol = V(:,1:i)^T *V(:,i+1);
        /*    cublasDgemv(cublas_handle_,
              CUBLAS_OP_T,
              n_,
              i+1,
              &one_,
              d_V_,
              n_,
              &d_V_[(i+1)*n_],
              1,
              &zero_,
              d_H_col_,
              1);*/

        vector_handler_->gemv("T", n_, i + 1, &one_, &zero_, d_V_,  &d_V_[(i+1)*n_], d_H_col_,"cuda");
        // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
        /*   cublasDgemv(cublas_handle_,
             CUBLAS_OP_N,
             n_,
             i+1,
             &minusone_,
             d_V_,
             n_,
             d_H_col_,
             1,
             &one_,
             &d_V_[n_*(i+1)],
             1);
             */
        vector_handler_->gemv("N", n_, i + 1, &one_, &minusone_, d_V_, d_H_col_,&d_V_[n_*(i+1)], "cuda" );  
        // copy H_col to aux, we will need it later

        cudaMemcpy(h_aux_, d_H_col_, sizeof(double) * (i+1), cudaMemcpyDeviceToHost);

        //Hcol = V(:,1:i)^T*V(:,i+1);
        /*   cublasDgemv(cublas_handle_,
             CUBLAS_OP_T,
             n_,
             i+1,
             &one_,
             d_V_,
             n_,
             &d_V_[(i+1)*n_],
             1,
             &zero_,
             d_H_col_,
             1);
             */

        vector_handler_-> gemv("T", n_, i + 1, &one_, &zero_, d_V_,  &d_V_[(i+1)*n_], d_H_col_,"cuda");
        // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol

        /*        cublasDgemv(cublas_handle_,
                  CUBLAS_OP_N,
                  n_,
                  i+1,
                  &minusone_,
                  d_V_,
                  n_,
                  d_H_col_,
                  1,
                  &one_,
                  &d_V_[n_*(i+1)],
                  1);
                  */

        vector_handler_->gemv("N", n_, i + 1, &one_, &minusone_, d_V_, d_H_col_,&d_V_[n_*(i+1)], "cuda" );  
        // copy H_col to H

        cudaMemcpy(&h_H_[i*(restart_+1)], d_H_col_, sizeof(double) * (i+1), cudaMemcpyDeviceToHost);
        // add both pieces together (unstable otherwise, careful here!!)
        for(int j=0; j<=i; ++j) {
          h_H_[i*(restart_+1)+j] += h_aux_[j]; 
        }
        t = 0.0;
        //    cublasDdot (cublas_handle_,  n_, &d_V_[(i+1)*n_], 1, &d_V_[(i+1)*n_], 1, &t);

        vec_w->setData( &d_V_[(i+1)*n_], "cuda");
        t = vector_handler_->dot(vec_w, vec_w, "cuda");  
        //set the last entry in Hessenberg matrix
        t=sqrt(t);
        h_H_[(i)*(restart_+1)+i+1] = t;    
        if(t != 0.0) {
          t = 1.0/t;
          //cublasDscal(cublas_handle_,n_,&t,&d_V_[(i+1)*n_], 1);
          vector_handler_->scal(&t, vec_w, "cuda");  
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
          return -1;
        }
        return 0;
        break;
        // the two low synch schemes
      case 2:
        // KS: the kernels are limited by the size of the shared memory on the GPU. If too many vectors in Krylov space, use standard cublas routines.
        // V[1:i]^T[V[i] w]
        /*    if(i < 200) {
              mass_inner_product_two_vectors(n_, i, &d_V_[i * n_],&d_V_[(i+1) * n_], d_V_, d_rvGPU_);
              } else {
              cublasDgemm(cublas_handle_,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              i + 1,//m
              2,//n
              n_,//k
              &one_,//alpha
              d_V_,//A
              n_,//lda
              &d_V_[i * n_],//B
              n_,//ldb
              &zero_,
              d_rvGPU_,//c
              i+1);//ldc 
              }*/
        vector_handler_->massDot2Vec(n_, d_V_, i, &d_V_[i * n_], d_rvGPU_, "cuda");
        // copy rvGPU to L
        cudaMemcpy(&h_L_[(i) * (restart_ + 1)], 
                   d_rvGPU_, 
                   (i + 1) * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(h_rv_, 
                   &d_rvGPU_[i + 1], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyDeviceToHost);

        for(int j=0; j<=i; ++j) {
          h_H_[(i)*(restart_+1)+j] = 0.0;
        }
        // triangular solve
        for(int j = 0; j <= i; ++j) {
          h_H_[(i) * (restart_ + 1) + j] = h_rv_[j];
          s = 0.0;
          for(int k = 0; k < j; ++k) {
            s += h_L_[j * (restart_ + 1) + k] * h_H_[(i) * (restart_ + 1) + k];
          } // for k
          h_H_[(i) * (restart_ + 1) + j] -= s; 
        }   // for j

        cudaMemcpy(d_Hcolumn_, 
                   &h_H_[(i) * (restart_ + 1)], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyHostToDevice);
        //again, use std cublas functions if Krylov space is too large
        /*     if(i < 200) {
               mass_axpy(n_, i, d_V_, &d_V_[(i+1) * n_],d_Hcolumn_);
               } else {
               cublasDgemm(cublas_handle_,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               n_,//m
               1,//n
               i + 1,//k
               &minusone_,//alpha
               d_V_,//A
               n_,//lda
               d_Hcolumn_,//B
               i + 1,//ldb
               &one_,
               &d_V_[(i + 1) * n_],//c
               n_);//ldc     
               }           */
        vector_handler_->massAxpy(n_, d_Hcolumn_, i, d_V_,  &d_V_[(i+1) * n_], "cuda");

        // normalize (second synch)
        vec_w->setData( &d_V_[(i+1)*n_], "cuda");
        t = vector_handler_->dot(vec_w, vec_w, "cuda");  
        //set the last entry in Hessenberg matrix
        t=sqrt(t);
        h_H_[(i)*(restart_+1)+i+1] = t;    
        if(t != 0.0) {
          t = 1.0/t;
          //cublasDscal(cublas_handle_,n_,&t,&d_V_[(i+1)*n_], 1);
          vector_handler_->scal(&t, vec_w, "cuda");  
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
          return -1;
        }
        return 0;
        break;

      case 3: //two synch Gauss-Seidel mgs, SUPER STABLE
        // according to unpublisjed work by ST
        // L is where we keep the triangular matrix(L is ON THE CPU)
        // if Krylov space is too large, use std cublas (because out of shared mmory)
        /*if(i < 200) {
          mass_inner_product_two_vectors(n_, i, &d_V_[i * n_],&d_V_[(i+1) * n_], d_V_, d_rvGPU_);
          } else {
          cublasDgemm(cublas_handle_,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          i + 1,//m
          2,//n
          n_,//k
          &one_,//alpha
          d_V_,//A
          n_,//lda
          &d_V_[i * n_],//B
          n_,//ldb
          &zero_,
          d_rvGPU_,//c
          i+1);//ldc 
          }*/

        vector_handler_->massDot2Vec(n_, d_V_, i, &d_V_[i * n_], d_rvGPU_, "cuda");
        // copy rvGPU to L
        cudaMemcpy(&h_L_[(i) * (restart_ + 1)], 
                   d_rvGPU_, 
                   (i + 1) * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(h_rv_, 
                   &d_rvGPU_[i + 1], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyDeviceToHost);

        for(int j = 0; j <= i; ++j) {
          h_H_[(i) * (restart_ + 1) + j] = 0.0;
        }
        //triangular solve
        for(int j = 0; j <= i; ++j) {
          h_H_[(i) * (restart_ + 1) + j] = h_rv_[j];
          s = 0.0;
          for(int k = 0; k < j; ++k) {
            s += h_L_[j * (restart_ + 1) + k] * h_H_[(i) * (restart_ + 1) + k];
          } // for k
          h_H_[(i) * (restart_ + 1) + j] -= s;
        }   // for j

        // now compute h_rv = L^T h_H
        double h;
        for(int j = 0; j <= i; ++j) {
          // go through COLUMN OF L
          h_rv_[j] = 0.0;
          for(int k = j + 1; k <= i; ++k) {
            h = h_L_[k * (restart_ + 1) + j];
            h_rv_[j] += h_H_[(i) * (restart_ + 1) + k] * h;
          }
        }

        // and do one more tri solve with L^T: h_aux = (I-L)^{-1}h_rv
        for(int j = 0; j <= i; ++j) {
          h_aux_[j] = h_rv_[j];
          s = 0.0;
          for(int k = 0; k < j; ++k) {
            s += h_L_[j * (restart_ + 1) + k] * h_aux_[k];
          } // for k
          h_aux_[j] -= s;
        }   // for j

        // and now subtract that from h_H
        for(int j=0; j<=i; ++j) {
          h_H_[(i)*(restart_+1)+j] -= h_aux_[j];
        }
        cudaMemcpy(d_Hcolumn_,
                   &h_H_[(i) * (restart_ + 1)], 
                   (i + 1) * sizeof(double), 
                   cudaMemcpyHostToDevice);
        // if Krylov space too large, use std cublas routines
        /*   if(i < 200) {
             mass_axpy(n_, i, d_V_, &d_V_[(i+1) * n_],d_Hcolumn_);
             } else {
             cublasDgemm(cublas_handle_,
             CUBLAS_OP_N,
             CUBLAS_OP_N,
             n_,//m
             1,//n
             i + 1,//k
             &minusone_,//alpha
             d_V_,//A
             n_,//lda
             d_Hcolumn_,//B
             i + 1,//ldb
             &one_,
             &d_V_[(i + 1) * n_],//c
             n_);//ldc     
             }*/

        vector_handler_->massAxpy(n_, d_Hcolumn_, i, d_V_,  &d_V_[(i+1) * n_], "cuda");
        // normalize (second synch)
        vec_w->setData( &d_V_[(i+1)*n_], "cuda");
        t = vector_handler_->dot(vec_w, vec_w, "cuda");  
        //set the last entry in Hessenberg matrix
        t=sqrt(t);
        h_H_[(i)*(restart_+1)+i+1] = t;    
        if(t != 0.0) {
          t = 1.0/t;
          //cublasDscal(cublas_handle_,n_,&t,&d_V_[(i+1)*n_], 1);
          vector_handler_->scal(&t, vec_w, "cuda");  
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
          return -1;
        }
        return 0;
        break;

      default:
        assert(0 && "Iterative refinement failed, wrong orthogonalization.\n");
        return -1;
        break;
    } // switch
  } // GramSchmidt


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


}
