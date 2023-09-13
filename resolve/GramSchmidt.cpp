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

        vec_rv_->setData(nullptr, "cuda");
        vec_rv_->setData(nullptr, "cpu");
        vec_Hcolumn_->setData(nullptr, "cuda");
        vec_Hcolumn_->setData(nullptr, "cpu");

        delete [] vec_rv_;    
        delete [] vec_Hcolumn_;;    
      }

      if(variant_ == cgs2) {
        delete h_aux_;
        vec_Hcolumn_->setData(nullptr, "cuda");
        //        vec_Hcolumn_->setData(nullptr, "cpu");
        delete [] vec_Hcolumn_;    
      }    
      if(variant_ == mgs_pm) {
        delete h_aux_;
      }

      vec_v_->setData(nullptr, "cuda");
      vec_v_->setData(nullptr, "cpu");
      vec_w_->setData(nullptr, "cuda");
      vec_w_->setData(nullptr, "cpu");

      delete [] vec_w_;
      delete [] vec_v_;   
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
        vec_rv_->allocate("cuda");      

        vec_Hcolumn_ = new vector_type(num_vecs_ + 1);
        vec_Hcolumn_->allocate("cuda");      
      }
      if(variant_ == cgs2) {
        h_aux_ = new real_type[num_vecs_ + 1];
        vec_Hcolumn_ = new vector_type(num_vecs_ + 1);
        vec_Hcolumn_->allocate("cuda");      
      }

      if(variant_ == mgs_pm) {
        h_aux_ = new real_type[num_vecs_ + 1];
      }
    }  

    return 0;
  }
  //this always happen on the GPU
  int GramSchmidt::orthogonalize(index_type n, vector::Vector* V, real_type* H, index_type i, std::string memspace)
  {

    if (memspace == "cuda") { // or hip

      double t;
      double s;

      switch (variant_){
        case mgs: 

          vec_w_->setData(V->getVectorData(i + 1, "cuda"), "cuda");
          for(int j = 0; j <= i; ++j) {
            t = 0.0;
            vec_v_->setData( V->getVectorData(j, "cuda"), "cuda");
            t = vector_handler_->dot(vec_v_, vec_w_, "cuda");  
            H[ idxmap(i, j, num_vecs_ + 1) ] = t; 
            t *= -1.0;
            vector_handler_->axpy(&t, vec_v_, vec_w_, "cuda");  
          }
          t = 0.0;
          t = vector_handler_->dot(vec_w_, vec_w_, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t; 
          if(fabs(t) > EPSILON) {
            t = 1.0/t;
            vector_handler_->scal(&t, vec_w_, "cuda");  
          } else {
            assert(0 && "Gram-Schmidt failed, vector with zero norm\n");
            return -1;
          }
          break;
        case cgs2:

          vec_v_->setData(V->getVectorData(i + 1, "cuda"), "cuda");
          vector_handler_->gemv("T", n, i + 1, &one, &zero, V,  vec_v_, vec_Hcolumn_,"cuda");

          // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
          vector_handler_->gemv("N", n, i + 1, &one, &minusone, V, vec_Hcolumn_, vec_v_, "cuda" );  

          // copy H_col to aux, we will need it later
          vec_Hcolumn_->setDataUpdated("cuda");
          vec_Hcolumn_->setCurrentSize(i + 1);
          vec_Hcolumn_->deepCopyVectorData(h_aux_, 0, "cpu");

          //Hcol = V(:,1:i)^T*V(:,i+1);
          vector_handler_->gemv("T", n, i + 1, &one, &zero, V,  vec_v_, vec_Hcolumn_,"cuda");

          // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
          vector_handler_->gemv("N", n, i + 1, &one, &minusone, V, vec_Hcolumn_, vec_v_, "cuda" );  

          // copy H_col to H
          vec_Hcolumn_->setDataUpdated("cuda");
          vec_Hcolumn_->deepCopyVectorData(&H[ idxmap(i, 0, num_vecs_ + 1)], 0, "cpu");

          // add both pieces together (unstable otherwise, careful here!!)
          t = 0.0;
          for(int j = 0; j <= i; ++j) {
            H[ idxmap(i, j, num_vecs_ + 1)] += h_aux_[j]; 
          }

          t = vector_handler_->dot(vec_v_, vec_v_, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t; 
          if(fabs(t) > EPSILON) {
            t = 1.0/t;
            vector_handler_->scal(&t, vec_v_, "cuda");  
          } else {
            assert(0 && "Gram-Schmidt failed, vector with zero norm\n");
            return -1;
          }
          return 0;
          break;
        case mgs_two_synch:
          // V[1:i]^T[V[i] w]
          vec_v_->setData(V->getVectorData(i, "cuda"), "cuda");
          vec_w_->setData(V->getVectorData(i + 1, "cuda"), "cuda");
          vec_rv_->setCurrentSize(i + 1);

          vector_handler_->massDot2Vec(n, V, i, vec_v_, vec_rv_, "cuda");
          vec_rv_->setDataUpdated("cuda");
          vec_rv_->copyData("cuda", "cpu");

          vec_rv_->deepCopyVectorData(&h_L_[idxmap(i, 0, num_vecs_ + 1)], 0, "cpu");
          h_rv_ = vec_rv_->getVectorData(1, "cpu");

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
          vec_Hcolumn_->update(&H[ idxmap(i, 0, num_vecs_ + 1)], "cpu", "cuda"); 
          vector_handler_->massAxpy(n, vec_Hcolumn_, i, V, vec_w_, "cuda");

          // normalize (second synch)
          t = vector_handler_->dot(vec_w_, vec_w_, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1)] = t;    
          if(fabs(t) > EPSILON) {
            t = 1.0 / t;
            vector_handler_->scal(&t, vec_w_, "cuda");  
          } else {
            assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
            return -1;
          }
          return 0;
          break;
        case mgs_pm:
          vec_v_->setData(V->getVectorData(i, "cuda"), "cuda");
          vec_w_->setData(V->getVectorData(i + 1, "cuda"), "cuda");
          vec_rv_->setCurrentSize(i + 1);

          vector_handler_->massDot2Vec(n, V, i, vec_v_, vec_rv_, "cuda");
          vec_rv_->setDataUpdated("cuda");
          vec_rv_->copyData("cuda", "cpu");

          vec_rv_->deepCopyVectorData(&h_L_[idxmap(i, 0, num_vecs_ + 1)], 0, "cpu");
          h_rv_ = vec_rv_->getVectorData(1, "cpu");

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
          vec_Hcolumn_->update(&H[ idxmap(i, 0, num_vecs_ + 1)], "cpu", "cuda"); 

          vector_handler_->massAxpy(n, vec_Hcolumn_, i, V,  vec_w_, "cuda");
          // normalize (second synch)
          t = vector_handler_->dot(vec_w_, vec_w_, "cuda");  
          //set the last entry in Hessenberg matrix
          t = sqrt(t);
          H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t;    
          if(fabs(t) > EPSILON) {
            t = 1.0 / t;
            vector_handler_->scal(&t, vec_w_, "cuda");  
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
