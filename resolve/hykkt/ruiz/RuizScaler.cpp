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
                           index_type total_n)
        : num_iterations_(num_iterations), n_(n), total_n_(total_n) {
      handler_ = new RuizScalingHandler(num_iterations, n, total_n);
    }

    RuizScaler::~RuizScaler() {
      delete handler_;
    }

    void RuizScaler::addHInfo(index_type* hes_i, index_type* hes_j, real_type* hes_v) {
      hes_i_ = hes_i;
      hes_j_ = hes_j;
      hes_v_ = hes_v;
    }

    void RuizScaler::addJInfo(index_type* jac_i, index_type* jac_j, real_type* jac_v) {
      jac_i_ = jac_i;
      jac_j_ = jac_j;
      jac_v_ = jac_v;
    }

    void RuizScaler::addJtInfo(index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v) {
      jac_tr_i_ = jac_tr_i;
      jac_tr_j_ = jac_tr_j;
      jac_tr_v_ = jac_tr_v;
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
  }
}