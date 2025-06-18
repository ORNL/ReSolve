#include <resolve/MemoryUtils.hpp>
#include "RuizScalingHandler.hpp"  

namespace ReSolve {
  using index_type = ReSolve::index_type;
  using real_type = ReSolve::real_type;
}

namespace  ReSolve {
  namespace hykkt {
    RuizScalingHandler::RuizScalingHandler(index_type num_iterations, index_type n, index_type totalN)
        : num_iterations_(num_iterations), n_(n), totalN_(totalN) 
    {
      cpuImpl = new RuizScalingHandlerCPU(num_iterations, n, totalN);

      #ifdef RE_SOLVE_USE_CUDA
      devImpl = new RuizScalingHandlerCUDA(num_iterations, n, totalN);
      #elseif RESOLVE_USE_HIP
      devImpl = new RuizScalingHandlerHIP(num_iterations, n, totalN);
      #else
      devImpl = nullptr;
      #endif
    }

    RuizScalingHandler::~RuizScalingHandler()
    {
      // Cleanup resources if necessary
      // This destructor can be used to free any allocated memory or resources
      // related to the scaling handler.
      delete cpuImpl;
      delete devImpl;
    }

    void RuizScalingHandler::scale(index_type hes_i, index_type hes_j, real_type* hes_v,
                                    index_type jac_i, index_type jac_j, real_type* jac_v,
                                    index_type jac_tr_i, index_type jac_tr_j, real_type* jac_tr_v,
                                    real_type* rhs1, real_type* rhs2,
                                    real_type* aggregate_scaling_vector,
                                    real_type* scaling_vector,
                                    memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST) {
        if (cpuImpl) {
          cpuImpl->scale(hes_i, hes_j, hes_v,
                         jac_i, jac_j, jac_v,
                         jac_tr_i, jac_tr_j, jac_tr_v,
                         rhs1, rhs2,
                         aggregate_scaling_vector,
                         scaling_vector);
        }
      } else if (memspace == memory::DEVICE) {
        if (devImpl) {
          devImpl->scale(hes_i, hes_j, hes_v,
                         jac_i, jac_j, jac_v,
                         jac_tr_i, jac_tr_j, jac_tr_v,
                         rhs1, rhs2,
                         aggregate_scaling_vector,
                         scaling_vector);
        }
      }
    }

    void RuizScalingHandler::reset(real_type* aggregate_scaling_vector, memory::MemorySpace memspace)
    {
      if (memspace == memory::HOST) {
        if (cpuImpl) {
          cpuImpl->reset(aggregate_scaling_vector);
        }
      } else if (memspace == memory::DEVICE) {
        if (devImpl) {
          devImpl->reset(aggregate_scaling_vector);
        }
      }
    }
  }
}