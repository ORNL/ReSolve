#include "RuizScaler.hpp"
#include "RuizScalingHandler.hpp"

namespace ReSolve {
  using index_type = ReSolve::index_type;
  using real_type = ReSolve::real_type;

  namespace hykkt {
    class RuizScaler;
    class RuizScalingHandler;
  }
}

namespace ReSolve {
  namespace hykkt {
    RuizScaler::RuizScaler(index_type num_iterations,
                           index_type n,
                           index_type totalN)
        : num_iterations_(num_iterations), n_(n), totalN_(totalN) {
      handler_ = new RuizScalingHandler(num_iterations, n, totalN);
    }

    RuizScaler::~RuizScaler() {
      delete handler_;
    }

    void RuizScaler::addHInfo(real_type* hes_v, index_type hes_i, index_type hes_j) {
      hes_v_ = hes_v;
      hes_i_ = hes_i;
      hes_j_ = hes_j;
    }

    void RuizScaler::addJInfo(real_type* jac_v, index_type jac_i, index_type jac_j) {
      jac_v_ = jac_v;
      jac_i_ = jac_i;
      jac_j_ = jac_j;
    }

    void RuizScaler::addJtInfo(real_type* jac_v, index_type jac_i, index_type jac_j) {
      jac_tr_v_ = jac_v;
      jac_tr_i_ = jac_i;
      jac_tr_j_ = jac_j;
    }

    void RuizScaler::addRhs1(real_type* rhs1) {
      rhs1_ = rhs1;
    }

    void RuizScaler::addRhs2(real_type* rhs2) {
      rhs2_ = rhs2;
    }

    real_type* RuizScaler::getAggregateScalingVector() const {
      return aggregate_scaling_vector_;
    }

    void RuizScaler::scale(memory::MemorySpace memspace) {
      handler_->scale(hes_i_, hes_j_, hes_v_,
                      jac_i_, jac_j_, jac_v_,
                      jac_tr_i_, jac_tr_j_, jac_tr_v_,
                      rhs1_, rhs2_,
                      scaling_vector_,
                      aggregate_scaling_vector_,
                      memspace);
    }

    void RuizScaler::reset(memory::MemorySpace memspace) {
      handler_->reset(aggregate_scaling_vector_, memspace);
    }
  }
}