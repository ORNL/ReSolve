#include <iostream>
#include <cassert>

#include <resolve/vector/Vector.hpp>
#include "GramSchmidt.hpp"

namespace ReSolve
{
  int idxmap(index_type i, index_type j, index_type col_lenght) {
    return  i * (col_lenght) + j;
  }
  GramSchmidt::GramSchmidt()
  {
    variant_ = mgs; //variant is enum now
    h_L_ = nullptr; 
    this->setup_complete_ = false;  
  }

  GramSchmidt::GramSchmidt(VectorHandler* vh,  GS_variant variant)
  {
    this->setVariant(variant);
    this->vector_handler_ = vh; h_L_ = nullptr; 
    h_L_ = nullptr; 
    this->setup_complete_ = false;  
  }

  GramSchmidt::~GramSchmidt()
  {
    if (setup_complete_) {
      if(variant_ == mgs_two_synch || variant_ == mgs_pm) {    
        delete h_L_;    
        delete h_rv_;    
        cudaFree(d_rvGPU_);    
        cudaFree(d_Hcolumn_);    
      }

      if(variant_ == cgs2) {
        delete h_aux_;
        cudaFree(d_H_col_);    
      }    
      if(variant_ == mgs_pm) {
        delete h_aux_;
      }
    }
  }

  int GramSchmidt::setVariant(GS_variant  variant)
  {
    if ((variant != mgs) && (variant != cgs2) && (variant != mgs_two_synch) && (variant != mgs_pm) && (variant != cgs)) { 
      this->variant_ = mgs;
      return 2;   
    }
    variant_ = variant;
    return 0;  
  }

  GS_variant GramSchmidt::getVariant()
  {
    return variant_;  
  }

  real_type* GramSchmidt::getL()
  {
    return h_L_;  
  }

  int GramSchmidt::setup(index_type restart)
  {
    if (setup_complete_) {
      return 1; // display some nasty comment too
    } else {
      this->num_vecs_ = restart;
      if(variant_ == mgs_two_synch || variant_ == mgs_pm) {
        h_L_  = new real_type[num_vecs_ * (num_vecs_ + 1)];
        h_rv_ = new real_type[num_vecs_ + 1];

        cudaMalloc(&(d_rvGPU_),   2 * (num_vecs_ + 1) * sizeof(real_type));
        cudaMalloc(&(d_Hcolumn_), 2 * (num_vecs_ + 1) * (num_vecs_ + 1) * sizeof(real_type));
      }
      if(variant_ == cgs2) {
        h_aux_ = new real_type[num_vecs_ + 1];
        cudaMalloc(&(d_H_col_), (num_vecs_ + 1) * sizeof(real_type));
      }

      if(variant_ == mgs_pm) {
        h_aux_ = new real_type[num_vecs_ + 1];
      }
    }  

    return 0;  
  }
  //this always happen on the GPU
  int GramSchmidt::orthogonalize(index_type n, real_type* V, real_type* H, index_type i, std::string memspace)
  {

    if (memspace == "cuda") { // or hip

      double t;
      double s;

      vector_type* vec_w = new vector_type(n);
      vector_type* vec_v = new vector_type(n);
      switch (variant_){
        case mgs: 

          vec_w->setData(&V[(i + 1) * n], "cuda");
          for(int j = 0; j <= i; ++j) {
            t = 0.0;
            vec_v->setData( &V[j * n], "cuda");
            t = vector_handler_->dot(vec_v, vec_w, "cuda");  
            H[ idxmap(i, j, num_vecs_ + 1) ] = t; 
            t *= -1.0;
            vector_handler_->axpy(&t, vec_v, vec_w, "cuda");  
          }
          t = 0.0;
          t = vector_handler_->dot(vec_w, vec_w, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t; 
          if(fabs(t) > EPSILON) {
            t = 1.0/t;
            vector_handler_->scal(&t, vec_w, "cuda");  
          } else {
            assert(0 && "Gram-Schmidt failed, vector with zero norm\n");
            return -1;
          }
          break;
        case cgs2:

          vector_handler_->gemv("T", n, i + 1, &one_, &zero_, V,  &V[(i + 1) * n], d_H_col_,"cuda");

          // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
          vector_handler_->gemv("N", n, i + 1, &one_, &minusone_, V, d_H_col_, &V[n * (i + 1)], "cuda" );  

          // copy H_col to aux, we will need it later
          cudaMemcpy(h_aux_, d_H_col_, sizeof(double) * (i + 1), cudaMemcpyDeviceToHost);

          //Hcol = V(:,1:i)^T*V(:,i+1);
          vector_handler_->gemv("T", n, i + 1, &one_, &zero_, V,  &V[(i + 1) * n], d_H_col_,"cuda");

          // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
          vector_handler_->gemv("N", n, i + 1, &one_, &minusone_, V, d_H_col_, &V[(i + 1) * n], "cuda" );  

          // copy H_col to H
          cudaMemcpy(&H[ idxmap(i, 0, num_vecs_ + 1)], d_H_col_, sizeof(double) * (i + 1), cudaMemcpyDeviceToHost);

          // add both pieces together (unstable otherwise, careful here!!)
          for(int j = 0; j <= i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1)] += h_aux_[j]; 
          }
          t = 0.0;

          vec_w->setData( &V[(i + 1) * n], "cuda");
          t = vector_handler_->dot(vec_w, vec_w, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t; 
          if(fabs(t) > EPSILON) {
            t = 1.0/t;
            vector_handler_->scal(&t, vec_w, "cuda");  
          } else {
            assert(0 && "Gram-Schmidt failed, vector with zero norm\n");
            return -1;
          }
          return 0;
          break;
        case mgs_two_synch:
          // V[1:i]^T[V[i] w]
          vector_handler_->massDot2Vec(n, V, i, &V[i * n], d_rvGPU_, "cuda");
          // copy rvGPU to L
          cudaMemcpy(&h_L_[idxmap(i, 0, num_vecs_ + 1)], 
                     d_rvGPU_, 
                     (i + 1) * sizeof(double),
                     cudaMemcpyDeviceToHost);

          cudaMemcpy(h_rv_, 
                     &d_rvGPU_[i + 1], 
                     (i + 1) * sizeof(double), 
                     cudaMemcpyDeviceToHost);

          for(int j=0; j<=i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1) ] = 0.0;
          }
          // triangular solve
          for(int j = 0; j <= i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1) ] = h_rv_[j];
            s = 0.0;
            for(int k = 0; k < j; ++k) {
              s += h_L_[ idxmap(j, k, num_vecs_ + 1) ] * H[ idxmap(i, k, num_vecs_ + 1) ];
            } // for k
            H[ idxmap(i, j, num_vecs_ + 1) ] -= s; 
          }   // for j

          cudaMemcpy(d_Hcolumn_, 
                     &H[ idxmap(i, 0, num_vecs_ + 1) ], 
                     (i + 1) * sizeof(double), 
                     cudaMemcpyHostToDevice);
          vector_handler_->massAxpy(n, d_Hcolumn_, i, V,  &V[(i + 1) * n], "cuda");

          // normalize (second synch)
          vec_w->setData( &V[(i + 1) * n], "cuda");
          t = vector_handler_->dot(vec_w, vec_w, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1)] = t;    
          if(fabs(t) > EPSILON) {
            t = 1.0 / t;
            vector_handler_->scal(&t, vec_w, "cuda");  
          } else {
            assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
            return -1;
          }
          return 0;
          break;
        case mgs_pm:
          vector_handler_->massDot2Vec(n, V, i, &V[i * n], d_rvGPU_, "cuda");
          // copy rvGPU to L
          cudaMemcpy(&h_L_[ idxmap(i, 0, num_vecs_ + 1) ], 
                     d_rvGPU_, 
                     (i + 1) * sizeof(double),
                     cudaMemcpyDeviceToHost);

          cudaMemcpy(h_rv_, 
                     &d_rvGPU_[i + 1], 
                     (i + 1) * sizeof(double), 
                     cudaMemcpyDeviceToHost);

          for(int j = 0; j <= i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1) ] = 0.0;
          }
          //triangular solve
          for(int j = 0; j <= i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1) ] = h_rv_[j];
            s = 0.0;
            for(int k = 0; k < j; ++k) {
              s += h_L_[ idxmap(j, k, num_vecs_ + 1) ] * H[ idxmap(i, k, num_vecs_ + 1) ];
            } // for k
            H[ idxmap(i, j, num_vecs_ + 1) ] -= s;
          }   // for j

          // now compute h_rv = L^T h_H
          double h;
          for(int j = 0; j <= i; ++j) {
            // go through COLUMN OF L
            h_rv_[j] = 0.0;
            for(int k = j + 1; k <= i; ++k) {
              h = h_L_[ idxmap(k, j, num_vecs_ + 1)];
              h_rv_[j] += H[ idxmap(i, k, num_vecs_ + 1) ] * h;
            }
          }

          // and do one more tri solve with L^T: h_aux = (I-L)^{-1}h_rv
          for(int j = 0; j <= i; ++j) {
            h_aux_[j] = h_rv_[j];
            s = 0.0;
            for(int k = 0; k < j; ++k) {
              s += h_L_[ idxmap(j, k, num_vecs_ + 1) ] * h_aux_[k];
            } // for k
            h_aux_[j] -= s;
          }   // for j

          // and now subtract that from h_H
          for(int j=0; j<=i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1) ] -= h_aux_[j];
          }
          cudaMemcpy(d_Hcolumn_,
                     &H[ idxmap(i, 0, num_vecs_ + 1) ], 
                     (i + 1) * sizeof(double), 
                     cudaMemcpyHostToDevice);

          vector_handler_->massAxpy(n, d_Hcolumn_, i, V,  &V[(i + 1) * n], "cuda");
          // normalize (second synch)
          vec_w->setData( &V[(i + 1) * n], "cuda");
          t = vector_handler_->dot(vec_w, vec_w, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t;    
          if(fabs(t) > EPSILON) {
            t = 1.0 / t;
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

      }//switch
    } else {
      std::cout<<"Not implemented (yet)"<<std::endl;
    }
  }//orthogonalize
}
