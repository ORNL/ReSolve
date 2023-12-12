#include <iostream>
#include <cassert>
#include <cmath>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include "GramSchmidt.hpp"

namespace ReSolve
{
  using out = io::Logger;

  int idxmap(index_type i, index_type j, index_type col_lenght) {
    return  i * (col_lenght) + j;
  }

  GramSchmidt::GramSchmidt()
  {
    variant_ = mgs; //variant is enum now
    h_L_ = nullptr; 
    this->setup_complete_ = false;  
  }

  GramSchmidt::GramSchmidt(VectorHandler* vh,  GSVariant variant)
  {
    this->setVariant(variant);
    this->vector_handler_ = vh;  
    h_L_ = nullptr; 
    this->setup_complete_ = false;  
  }

  GramSchmidt::~GramSchmidt()
  {
    if (setup_complete_) {
      if(variant_ == mgs_two_synch || variant_ == mgs_pm) {    
        delete h_L_;    
        delete h_rv_;    

        delete vec_rv_;    
        delete vec_Hcolumn_;;    
      }

      if(variant_ == cgs2) {
        delete h_aux_;
        delete vec_Hcolumn_;    
      }    
      if(variant_ == mgs_pm) {
        delete h_aux_;
      }

      delete vec_w_;
      delete vec_v_;   
    }
  }

  int GramSchmidt::setVariant(GSVariant  variant)
  {
    if ((variant != mgs) && (variant != cgs2) && (variant != mgs_two_synch) && (variant != mgs_pm) && (variant != cgs1)) { 
      this->variant_ = mgs;
      return 2;   
    }
    variant_ = variant;
    return 0;  
  }

  GramSchmidt::GSVariant GramSchmidt::getVariant()
  {
    return variant_;  
  }

  real_type* GramSchmidt::getL()
  {
    return h_L_;  
  }

  bool GramSchmidt::isSetupComplete()
  {
    return setup_complete_;
  }

  int GramSchmidt::setup(index_type n, index_type restart)
  {
    if (setup_complete_) {
      return 1; // display some nasty comment too
    } else {
      vec_w_ = new vector_type(n);
      vec_v_ = new vector_type(n);
      this->num_vecs_ = restart;
      if(variant_ == mgs_two_synch || variant_ == mgs_pm) {
        h_L_  = new real_type[num_vecs_ * (num_vecs_ + 1)];
        h_rv_ = new real_type[num_vecs_ + 1];

        vec_rv_ = new vector_type(num_vecs_ + 1, 2);
        vec_rv_->allocate(memory::DEVICE);      

        vec_Hcolumn_ = new vector_type(num_vecs_ + 1);
        vec_Hcolumn_->allocate(memory::DEVICE);      
      }
      if(variant_ == cgs2) {
        h_aux_ = new real_type[num_vecs_ + 1];
        vec_Hcolumn_ = new vector_type(num_vecs_ + 1);
        vec_Hcolumn_->allocate(memory::DEVICE);      
      }

      if(variant_ == mgs_pm) {
        h_aux_ = new real_type[num_vecs_ + 1];
      }
    }  

    return 0;
  }

  //this always happen on the GPU
  int GramSchmidt::orthogonalize(index_type n, vector::Vector* V, real_type* H, index_type i, memory::MemorySpace memspace)
  {
    using namespace constants;

    if (memspace == memory::DEVICE) {

      double t;
      double s;

      switch (variant_){
        case mgs: 

          vec_w_->setData(V->getVectorData(i + 1, memory::DEVICE), memory::DEVICE);
          for(int j = 0; j <= i; ++j) {
            t = 0.0;
            vec_v_->setData( V->getVectorData(j, memory::DEVICE), memory::DEVICE);
            t = vector_handler_->dot(vec_v_, vec_w_, memspace);  
            H[ idxmap(i, j, num_vecs_ + 1) ] = t; 
            t *= -1.0;
            vector_handler_->axpy(&t, vec_v_, vec_w_, memspace);  
          }
          t = 0.0;
          t = vector_handler_->dot(vec_w_, vec_w_, memspace);
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t; 
          if(fabs(t) > EPSILON) {
            t = 1.0/t;
            vector_handler_->scal(&t, vec_w_, memspace);  
          } else {
            assert(0 && "Gram-Schmidt failed, vector with ZERO norm\n");
            return -1;
          }
          break;
        case cgs2:

          vec_v_->setData(V->getVectorData(i + 1, memory::DEVICE), memory::DEVICE);
          vector_handler_->gemv('T', n, i + 1, &ONE, &ZERO, V,  vec_v_, vec_Hcolumn_, memspace);
          // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
          vector_handler_->gemv('N', n, i + 1, &ONE, &MINUSONE, V, vec_Hcolumn_, vec_v_, memspace );  
          mem_.deviceSynchronize();
          
          // copy H_col to aux, we will need it later
          vec_Hcolumn_->setDataUpdated(memory::DEVICE);
          vec_Hcolumn_->setCurrentSize(i + 1);
          vec_Hcolumn_->deepCopyVectorData(h_aux_, 0, memory::HOST);
          mem_.deviceSynchronize();

          //Hcol = V(:,1:i)^T*V(:,i+1);
          vector_handler_->gemv('T', n, i + 1, &ONE, &ZERO, V,  vec_v_, vec_Hcolumn_, memspace);
          mem_.deviceSynchronize();

          // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
          vector_handler_->gemv('N', n, i + 1, &ONE, &MINUSONE, V, vec_Hcolumn_, vec_v_, memspace );  
          mem_.deviceSynchronize();

          // copy H_col to H
          vec_Hcolumn_->setDataUpdated(memory::DEVICE);
          vec_Hcolumn_->deepCopyVectorData(&H[ idxmap(i, 0, num_vecs_ + 1)], 0, memory::HOST);
          mem_.deviceSynchronize();

          // add both pieces together (unstable otherwise, careful here!!)
          t = 0.0;
          for(int j = 0; j <= i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1)] += h_aux_[j];
          }

          t = vector_handler_->dot(vec_v_, vec_v_, memspace);  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t; 

          if(fabs(t) > EPSILON) {
            t = 1.0/t;
            vector_handler_->scal(&t, vec_v_, memspace);  
          } else {
            assert(0 && "Gram-Schmidt failed, vector with ZERO norm\n");
            return -1;
          }
          return 0;
          break;
        case mgs_two_synch:
          // V[1:i]^T[V[i] w]
          vec_v_->setData(V->getVectorData(i, memory::DEVICE), memory::DEVICE);
          vec_w_->setData(V->getVectorData(i + 1, memory::DEVICE), memory::DEVICE);
          vec_rv_->setCurrentSize(i + 1);

          vector_handler_->massDot2Vec(n, V, i, vec_v_, vec_rv_, memspace);
          vec_rv_->setDataUpdated(memory::DEVICE);
          vec_rv_->copyData(memory::DEVICE, memory::HOST);

          vec_rv_->deepCopyVectorData(&h_L_[idxmap(i, 0, num_vecs_ + 1)], 0, memory::HOST);
          h_rv_ = vec_rv_->getVectorData(1, memory::HOST);

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
          vec_Hcolumn_->setCurrentSize(i + 1);
          vec_Hcolumn_->update(&H[ idxmap(i, 0, num_vecs_ + 1)], memory::HOST, memory::DEVICE); 
          vector_handler_->massAxpy(n, vec_Hcolumn_, i, V, vec_w_, memspace);

          // normalize (second synch)
          t = vector_handler_->dot(vec_w_, vec_w_, memspace);  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1)] = t;    
          if(fabs(t) > EPSILON) {
            t = 1.0 / t;
            vector_handler_->scal(&t, vec_w_, memspace);  
          } else {
            assert(0 && "Iterative refinement failed, Krylov vector with ZERO norm\n");
            return -1;
          }
          return 0;
          break;
        case mgs_pm:
          vec_v_->setData(V->getVectorData(i, memory::DEVICE), memory::DEVICE);
          vec_w_->setData(V->getVectorData(i + 1, memory::DEVICE), memory::DEVICE);
          vec_rv_->setCurrentSize(i + 1);

          vector_handler_->massDot2Vec(n, V, i, vec_v_, vec_rv_, memspace);
          vec_rv_->setDataUpdated(memory::DEVICE);
          vec_rv_->copyData(memory::DEVICE, memory::HOST);

          vec_rv_->deepCopyVectorData(&h_L_[idxmap(i, 0, num_vecs_ + 1)], 0, memory::HOST);
          h_rv_ = vec_rv_->getVectorData(1, memory::HOST);

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

          vec_Hcolumn_->setCurrentSize(i + 1);
          vec_Hcolumn_->update(&H[ idxmap(i, 0, num_vecs_ + 1)], memory::HOST, memory::DEVICE); 

          vector_handler_->massAxpy(n, vec_Hcolumn_, i, V,  vec_w_, memspace);
          // normalize (second synch)
          t = vector_handler_->dot(vec_w_, vec_w_, memspace);  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t;    
          if(fabs(t) > EPSILON) {
            t = 1.0 / t;
            vector_handler_->scal(&t, vec_w_, memspace);  
          } else {
            assert(0 && "Iterative refinement failed, Krylov vector with ZERO norm\n");
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
      out::error() << "Not implemented (yet)" << std::endl;
      return -1;
    }
    return 0;
  }//orthogonalize
}
