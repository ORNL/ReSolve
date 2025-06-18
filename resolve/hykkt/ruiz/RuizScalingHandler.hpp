#include <resolve/MemoryUtils.hpp>

namespace ReSolve {
  using index_type = ReSolve::index_type;
  using real_type = ReSolve::real_type;
}

namespace  ReSolve {
  namespace hykkt {
    class RuizScalingHandler {
      public:
        RuizScalingHandler(index_type num_iterations, index_type n, index_type totalN);
        ~RuizScalingHandler();

        void scale(index_type hes_i, index_type hes_j, real_type* hes_v,
                   index_type jac_i, index_type jac_j, real_type* jac_v,
                   index_type jac_tr_i, index_type jac_tr_j, real_type* jac_tr_v,
                   real_type* rhs1, real_type* rhs2,
                   real_type* aggregate_scaling_vector,
                   real_type* scaling_vector,
                   memory::MemorySpace memspace);
                   
        void reset(real_type* aggregate_scaling_vector, memory::MemorySpace memspace);
      private:
        index_type num_iterations_;
        index_type n_;
        index_type totalN_;

        RuizScalingHandlerImpl* cpuImpl;
        RuizScalingHandlerImpl* devImpl;
    };
  }
}