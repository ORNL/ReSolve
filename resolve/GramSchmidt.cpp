#include <iostream>
#include <cassert>
#include <cmath>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include "GramSchmidt.hpp"

namespace ReSolve
{
  using out = io::Logger;

  index_type idxmap(index_type i, index_type j, index_type col_length)
  {
    return  i * (col_length) + j;
  }

  GramSchmidt::GramSchmidt(VectorHandler* vh,  GSVariant variant)
    : variant_(variant),
      setup_complete_(false),
      vector_handler_(vh)
  {
    if (vector_handler_->getIsCudaEnabled() || vector_handler_->getIsHipEnabled()) {
      memspace_ = memory::DEVICE;
    } else {
      memspace_ = memory::HOST;
    }
  }

  GramSchmidt::~GramSchmidt()
  {
    if (setup_complete_) {
      freeGramSchmidtData();
    }
  }

  /**
   * @brief Sets/changes Gram-Schmidt variant
   *
   * This function should leave Gram-Schmidt class instance in the same state
   * as it found it only with different implementation of the orthogonalization.
   *
   * @param[in] variant - Gram-Schmidt orthogonalization variant
   */
  int GramSchmidt::setVariant(GSVariant variant)
  {
    // If the same variant is already set, do nothing.
    if(variant == variant_) {
      return 0;
    }

    // If Gram-Scmidt data is not allocated, just set the variant and exit.
    if (!setup_complete_) {
      variant_ = variant;
      return 0;
    }

    // If we reached this point, the setup was done for a different variant.
    index_type n = vec_v_->getSize();

    // First delete current data structures
    freeGramSchmidtData();

    // Next, change variant and set up Gram-Scmidt again
    variant_ = variant;
    setup(n, num_vecs_);

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
      if ((vec_v_->getSize() != n) || (num_vecs_ != restart)) {
        freeGramSchmidtData();
      }
    }

    vec_w_ = new vector_type(n);
    vec_v_ = new vector_type(n);
    vec_x_ = new vector_type(n, 2); // n x 2 multivector view

    num_vecs_ = restart;
    if((variant_ == MGS_TWO_SYNC) || (variant_ == MGS_PM)) {
      h_L_  = new real_type[num_vecs_ * (num_vecs_ + 1)]();

      vec_rv_ = new vector_type(num_vecs_ + 1, 2);
      vec_rv_->allocate(memspace_);
      vec_rv_->setToZero(memspace_);

      vec_Hcolumn_ = new vector_type(num_vecs_ + 1);
      vec_Hcolumn_->allocate(memspace_);
      vec_Hcolumn_->setToZero(memspace_);
    }
    if(variant_ == CGS2) {
      h_aux_ = new real_type[num_vecs_ + 1]();
      vec_Hcolumn_ = new vector_type(num_vecs_ + 1);
      vec_Hcolumn_->allocate(memspace_);
      vec_Hcolumn_->setToZero(memspace_);
    }
    if(variant_ == CGS1) {
      vec_Hcolumn_ = new vector_type(num_vecs_ + 1);
      vec_Hcolumn_->allocate(memspace_);
      vec_Hcolumn_->setToZero(memspace_);
    }
    if(variant_ == MGS_PM) {
      h_aux_ = new real_type[num_vecs_ + 1]();
    }

    setup_complete_ = true;
    return 0;
  }

  int GramSchmidt::orthogonalize(index_type n, vector::Vector* V, real_type* H, index_type i)
  {
    using namespace constants;

    double t = 0.0;
    double s = 0.0;
    real_type* h_rv = nullptr;

    switch (variant_) {
      case MGS:
        vec_w_->setData(V->getData(i + 1, memspace_), memspace_);
        for(int j = 0; j <= i; ++j) {
          t = 0.0;
          vec_v_->setData( V->getData(j, memspace_), memspace_);
          t = vector_handler_->dot(vec_v_, vec_w_, memspace_);
          H[ idxmap(i, j, num_vecs_ + 1) ] = t;
          t *= -1.0;
          vector_handler_->axpy(&t, vec_v_, vec_w_, memspace_);
        }
        t = 0.0;
        t = vector_handler_->dot(vec_w_, vec_w_, memspace_);
        //set the last entry in Hessenberg matrix
        t = std::sqrt(t);
        H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t;
        if(std::abs(t) > MACHINE_EPSILON) {
          t = 1.0/t;
          vector_handler_->scal(&t, vec_w_, memspace_);
        } else {
          assert(0 && "Gram-Schmidt failed, vector with ZERO norm\n");
          return 1;
        }
        return 0;

      case CGS2:
        vec_v_->setData(V->getData(i + 1, memspace_), memspace_);
        vector_handler_->gemv('T', n, i + 1, &ONE, &ZERO, V,  vec_v_, vec_Hcolumn_, memspace_);
        // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
        vector_handler_->gemv('N', n, i + 1, &ONE, &MINUS_ONE, V, vec_Hcolumn_, vec_v_, memspace_ );
        mem_.deviceSynchronize();

        // copy H_col to aux, we will need it later
        vec_Hcolumn_->setDataUpdated(memspace_);
        vec_Hcolumn_->resize(i + 1);
        vec_Hcolumn_->copyDataTo(h_aux_, 0, memory::HOST);
        mem_.deviceSynchronize();

        //Hcol = V(:,1:i)^T*V(:,i+1);
        vector_handler_->gemv('T', n, i + 1, &ONE, &ZERO, V,  vec_v_, vec_Hcolumn_, memspace_);
        mem_.deviceSynchronize();

        // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
        vector_handler_->gemv('N', n, i + 1, &ONE, &MINUS_ONE, V, vec_Hcolumn_, vec_v_, memspace_ );
        mem_.deviceSynchronize();

        // copy H_col to H
        vec_Hcolumn_->setDataUpdated(memspace_);
        vec_Hcolumn_->copyDataTo(&H[ idxmap(i, 0, num_vecs_ + 1)], 0, memory::HOST);
        mem_.deviceSynchronize();

        // add both pieces together (unstable otherwise, careful here!!)
        t = 0.0;
        for(int j = 0; j <= i; ++j) {
          H[ idxmap(i, j, num_vecs_ + 1)] += h_aux_[j];
        }

        t = vector_handler_->dot(vec_v_, vec_v_, memspace_);
        //set the last entry in Hessenberg matrix
        t = std::sqrt(t);
        H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t;

        if(std::abs(t) > MACHINE_EPSILON) {
          t = 1.0/t;
          vector_handler_->scal(&t, vec_v_, memspace_);
        } else {
          assert(0 && "Gram-Schmidt failed, vector with ZERO norm\n");
          return 1;
        }
        return 0;

      case MGS_TWO_SYNC:
        // V[1:i]^T[V[i] w]
        vec_x_->setData(V->getData(i, memspace_), memspace_);
        vec_w_->setData(V->getData(i + 1, memspace_), memspace_);
        vec_rv_->resize(i + 1);

        vector_handler_->massDot2Vec(n, V, i + 1, vec_x_, vec_rv_, memspace_);
        vec_rv_->setDataUpdated(memspace_);
        if (memspace_ == memory::DEVICE) {
          vec_rv_->syncData(memory::HOST);
        }

        vec_rv_->copyDataTo(&h_L_[idxmap(i, 0, num_vecs_ + 1)], 0, memory::HOST);
        h_rv = vec_rv_->getData(1, memory::HOST);

        for(int j=0; j<=i; ++j) {
          H[ idxmap(i, j, num_vecs_ + 1) ] = 0.0;
        }
        // triangular solve
        for(int j = 0; j <= i; ++j) {
          H[ idxmap(i, j, num_vecs_ + 1) ] = h_rv[j];
          s = 0.0;
          for(int k = 0; k < j; ++k) {
            s += h_L_[ idxmap(j, k, num_vecs_ + 1) ] * H[ idxmap(i, k, num_vecs_ + 1) ];
          } // for k
          H[ idxmap(i, j, num_vecs_ + 1) ] -= s;
        }   // for j
        vec_Hcolumn_->resize(i + 1);
        vec_Hcolumn_->copyDataFrom(&H[ idxmap(i, 0, num_vecs_ + 1)], memory::HOST, memspace_);
        vector_handler_->massAxpy(n, vec_Hcolumn_, i + 1, V, vec_w_, memspace_);

        // normalize (second synch)
        t = vector_handler_->dot(vec_w_, vec_w_, memspace_);
        //set the last entry in Hessenberg matrix
        t = std::sqrt(t);
        H[ idxmap(i, i + 1, num_vecs_ + 1)] = t;
        if(std::abs(t) > MACHINE_EPSILON) {
          t = 1.0 / t;
          vector_handler_->scal(&t, vec_w_, memspace_);
          for (int ii=0; ii<=i; ++ii)
          {
            vec_v_->setData(V->getData(ii, memspace_), memspace_);
            vec_w_->setData(V->getData(i + 1, memspace_), memspace_);
          }
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with ZERO norm\n");
          return 1;
        }
        h_rv = nullptr;
        return 0;

      case MGS_PM:
        vec_x_->setData(V->getData(i, memspace_), memspace_);
        vec_w_->setData(V->getData(i + 1, memspace_), memspace_);
        vec_rv_->resize(i + 1);

        vector_handler_->massDot2Vec(n, V, i + 1, vec_x_, vec_rv_, memspace_);
        vec_rv_->setDataUpdated(memspace_);
        if (memspace_ == memory::DEVICE) {
          vec_rv_->syncData(memory::HOST);
        }

        vec_rv_->copyDataTo(&h_L_[idxmap(i, 0, num_vecs_ + 1)], 0, memory::HOST);
        h_rv = vec_rv_->getData(1, memory::HOST);

        for(int j = 0; j <= i; ++j) {
          H[ idxmap(i, j, num_vecs_ + 1) ] = 0.0;
        }

        //triangular solve
        for(int j = 0; j <= i; ++j) {
          H[ idxmap(i, j, num_vecs_ + 1) ] = h_rv[j];
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
          h_rv[j] = 0.0;
          for(int k = j + 1; k <= i; ++k) {
            h = h_L_[ idxmap(k, j, num_vecs_ + 1)];
            h_rv[j] += H[ idxmap(i, k, num_vecs_ + 1) ] * h;
          }
        }

        // and do one more tri solve with L^T: h_aux = (I-L)^{-1}h_rv
        for(int j = 0; j <= i; ++j) {
          h_aux_[j] = h_rv[j];
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

        vec_Hcolumn_->resize(i + 1);
        vec_Hcolumn_->copyDataFrom(&H[ idxmap(i, 0, num_vecs_ + 1)], memory::HOST, memspace_);

        vector_handler_->massAxpy(n, vec_Hcolumn_, i + 1, V,  vec_w_, memspace_);
        // normalize (second synch)
        t = vector_handler_->dot(vec_w_, vec_w_, memspace_);
        //set the last entry in Hessenberg matrix
        t = std::sqrt(t);
        H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t;
        if(std::abs(t) > MACHINE_EPSILON) {
          t = 1.0 / t;
          vector_handler_->scal(&t, vec_w_, memspace_);
        } else {
          assert(0 && "Iterative refinement failed, Krylov vector with ZERO norm\n");
          return 1;
        }
        h_rv = nullptr;
        return 0;

      case CGS1:
        vec_v_->setData(V->getData(i + 1, memspace_), memspace_);
        //Hcol = V(:,1:i)^T*V(:,i+1);
        vector_handler_->gemv('T', n, i + 1, &ONE, &ZERO, V,  vec_v_, vec_Hcolumn_, memspace_);
        // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
        vector_handler_->gemv('N', n, i + 1, &ONE, &MINUS_ONE, V, vec_Hcolumn_, vec_v_, memspace_ );
        mem_.deviceSynchronize();

        // copy H_col to H
        vec_Hcolumn_->setDataUpdated(memspace_);
        vec_Hcolumn_->resize(i + 1);
        vec_Hcolumn_->copyDataTo(&H[ idxmap(i, 0, num_vecs_ + 1)], 0, memory::HOST);
        mem_.deviceSynchronize();

        t = vector_handler_->dot(vec_v_, vec_v_, memspace_);
        //set the last entry in Hessenberg matrix
        t = std::sqrt(t);
        H[ idxmap(i, i + 1, num_vecs_ + 1) ] = t;
        if(std::abs(t) > MACHINE_EPSILON) {
          t = 1.0/t;
          vector_handler_->scal(&t, vec_v_, memspace_);
        } else {
          assert(0 && "Gram-Schmidt failed, vector with ZERO norm\n");
          return 1;
        }
        return 0;

      default:
        assert(0 && "Iterative refinement failed, wrong orthogonalization.\n");
        return 1;
    } //switch

    return 0;
  } // int orthogonalize()

  //
  // Private methods
  //

  int GramSchmidt::freeGramSchmidtData()
  {
    if(variant_ == MGS_TWO_SYNC || variant_ == MGS_PM) {
      delete[] h_L_;
      h_L_ = nullptr;

      delete vec_rv_;
      vec_rv_ = nullptr;
      delete vec_Hcolumn_;
      vec_Hcolumn_ = nullptr;
    }

    if (variant_ == CGS2) {
      delete[] h_aux_;
      h_aux_ = nullptr;
      delete vec_Hcolumn_;
      vec_Hcolumn_ = nullptr;
    }

    if (variant_ == CGS1) {
      delete vec_Hcolumn_;
      vec_Hcolumn_ = nullptr;
    }

    if (variant_ == MGS_PM) {
      delete[] h_aux_;
      h_aux_ = nullptr;
    }

    delete vec_w_;
    vec_w_ = nullptr;
    delete vec_v_;
    vec_v_ = nullptr;

    setup_complete_ = false;
    return 0;
  }


} // namespace ReSolve
