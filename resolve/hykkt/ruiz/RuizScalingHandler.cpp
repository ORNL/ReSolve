#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include "RuizScalingHandler.hpp"

namespace  ReSolve {
  namespace hykkt {
    RuizScalingHandler::RuizScalingHandler(index_type num_iterations, index_type n, index_type total_n,
                                           memory::MemorySpace memspace)
        : num_iterations_(num_iterations), n_(n), total_n_(total_n), memspace_(memspace)
    {
      #ifdef RESOLVE_USE_CUDA
      handlerImpl_ = new RuizScalingKernelsCUDA(num_iterations, n, total_n);
      #elseif RESOLVE_USE_HIP
      handlerImpl_ = new RuizScalingKernelsHIP(num_iterations, n, total_n);
      #else RESOLVE_USE_HIP
      handlerImpl_ = new RuizScalingKernelsCPU(num_iterations, n, total_n);
      #endif
    }

    RuizScalingHandler::~RuizScalingHandler()
    {
      delete handlerImpl_;
    }

    void RuizScalingHandler::scale(index_type* hes_i, index_type* hes_j, real_type* hes_v,
                                    index_type* jac_i, index_type* jac_j, real_type* jac_v,
                                    index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v,
                                    real_type* rhs1, real_type* rhs2,
                                    real_type* aggregate_scaling_vector,
                                    real_type* scaling_vector)
    {
      using namespace ReSolve::constants;

      for (index_type i = 0; i < num_iterations_; ++i) {
        handlerImpl_->adaptRowMax(n_, total_n_,
                          hes_i, hes_j, hes_v,
                          jac_i, jac_j, jac_v,
                          jac_tr_i, jac_tr_j, jac_tr_v,
                          rhs1, rhs2,
                          aggregate_scaling_vector,
                          scaling_vector);

        handlerImpl_->adaptDiagScale(n_, total_n_,
                              hes_i, hes_j, hes_v,
                              jac_i, jac_j, jac_v,
                              jac_tr_i, jac_tr_j, jac_tr_v,
                              rhs1, rhs2,
                              aggregate_scaling_vector,
                              scaling_vector);
      }
    }
  }
}