#include <resolve/MemoryUtils.hpp>
#include "RuizScalingKernelImpl.hpp"

namespace  ReSolve {
  namespace hykkt {
    class RuizScalingHandler {
      public:
        RuizScalingHandler(index_type num_iterations, index_type n, index_type total_n, memory::MemorySpace memspace);
        ~RuizScalingHandler();

        void scale(index_type* hes_i, index_type* hes_j, real_type* hes_v,
                   index_type* jac_i, index_type* jac_j, real_type* jac_v,
                   index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v,
                   real_type* rhs1, real_type* rhs2,
                   real_type* aggregate_scaling_vector,
                   real_type* scaling_vector);
      private:
        index_type num_iterations_;
        index_type n_;
        index_type total_n_;

        RuizScalingKernelImpl* handlerImpl_;

        memory::MemorySpace memspace_;
        MemoryHandler mem_;
    };
  }
}