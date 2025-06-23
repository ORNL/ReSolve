#include "RuizScaler.hpp"
#include "RuizScalingKernelsCPU.hpp"
#ifdef RESOLVE_USE_CUDA
#include "RuizScalingKernelsCUDA.hpp"
#endif
#ifdef RESOLVE_USE_HIP
#include "RuizScalingKernelsHIP.hpp"
#endif

namespace ReSolve {
  using index_type = ReSolve::index_type;
  using real_type = ReSolve::real_type;

  using namespace ReSolve::constants;

  namespace hykkt {
    class RuizScaler;
    class RuizScalingHandler;
  }
}

namespace ReSolve {
  namespace hykkt {
    RuizScaler::RuizScaler(index_type num_iterations,
                           index_type n,
                           index_type total_n,
                           memory::MemorySpace memspace)
        : num_iterations_(num_iterations), n_(n), total_n_(total_n), memspace_(memspace)
    {
      if (memspace_ == memory::HOST) {
        kernelImpl_ = new RuizScalingKernelsCPU();
      } else {
#ifdef RESOLVE_USE_CUDA
      kernelImpl_ = new RuizScalingKernelsCUDA();
#else
      kernelImpl_ = new RuizScalingKernelsHIP();
#endif
      }


      allocateWorkspace();
    }

    RuizScaler::~RuizScaler()
    {
      deallocateWorkspace();
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

    void RuizScaler::scale() {
      resetScaling();

      for (index_type i = 0; i < num_iterations_; ++i) {
        kernelImpl_->adaptRowMax(n_, total_n_,
                                  hes_i_, hes_j_, hes_v_,
                                  jac_i_, jac_j_, jac_v_,
                                  jac_tr_i_, jac_tr_j_, jac_tr_v_,
                                  rhs1_, rhs2_,
                                  aggregate_scaling_vector_,
                                  scaling_vector_);

        kernelImpl_->adaptDiagScale(n_, total_n_,
                                    hes_i_, hes_j_, hes_v_,
                                    jac_i_, jac_j_, jac_v_,
                                    jac_tr_i_, jac_tr_j_, jac_tr_v_,
                                    rhs1_, rhs2_,
                                    aggregate_scaling_vector_,
                                    scaling_vector_);
      }
    }

    void RuizScaler::resetScaling()
    {
      if (memspace_ == memory::HOST) {
        mem_.setArrayToConstOnHost(aggregate_scaling_vector_, ONE, total_n_);
      } else {
        mem_.setArrayToConstOnDevice(aggregate_scaling_vector_, ONE, total_n_);
      }
    }

    void RuizScaler::allocateWorkspace()
    {
      if (memspace_ == memory::HOST) {
        scaling_vector_ = new real_type[n_];
        aggregate_scaling_vector_ = new real_type[total_n_];
      } else {
        mem_.allocateArrayOnDevice(&scaling_vector_, total_n_);
        mem_.allocateArrayOnDevice(&aggregate_scaling_vector_, total_n_);
      }
      resetScaling();
    }

    void RuizScaler::deallocateWorkspace() {
      if (memspace_ == memory::HOST) {
        delete[] scaling_vector_;
        delete[] aggregate_scaling_vector_;
      } else {
        mem_.deleteOnDevice(scaling_vector_);
        mem_.deleteOnDevice(aggregate_scaling_vector_);
      }
    }
  }
}