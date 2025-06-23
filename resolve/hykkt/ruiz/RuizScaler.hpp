#include <resolve/MemoryUtils.hpp>
#include <RuizScalingHandler.hpp>

namespace ReSolve {
  using index_type = ReSolve::index_type;
  using real_type = ReSolve::real_type;

  namespace hykkt {
    class RuizScalingHandler;
  }
}

namespace ReSolve {
  namespace hykkt {
    class RuizScaler {
      public:
        RuizScaler(index_type num_iterations,
                    index_type n,
                    index_type total_n);
        ~RuizScaler();

        /**
         *  @brief Add Hessian information to the Ruiz scaling handler.
         *  @param hes_i[in] - Row data of the Hessian matrix.
         *  @param hes_j[in] - Column data of the Hessian matrix.
         *  @param hes_v[in] - Pointer to the Hessian nonzero values.
         */
        void addHInfo(index_type* hes_i, index_type* hes_j, real_type* hes_v);

        /**
         *  @brief Add Jacobian information to the Ruiz scaling handler.
         *  @param jac_i[in] - Row data of the Jacobian matrix.
         *  @param jac_j[in] - Column data of the Jacobian matrix.
         *  @param jac_v[in] - Pointer to the Jacobian nonzero values.
         */
        void addJInfo(index_type* jac_i, index_type* jac_j, real_type* jac_v);

        /**
         *  @brief Add Jacobian transpose information to the Ruiz scaling handler.
         *  @param jac_i[in] - Row data of the Jacobian transpose matrix.
         *  @param jac_j[in] - Column data of the Jacobian transpose matrix.
         *  @param jac_v[in] - Pointer to the Jacobian transpose nonzero values.
         */
        void addJtInfo(index_type* jac_i, index_type* jac_j, real_type* jac_v);

        /**
         *  @brief Add right-hand side vector to the Ruiz scaling handler.
         *  @param rhs1[in] - Pointer to the top right-hand side vector.
         */
        void addRhs1(real_type* rhs1);

        /**
         *  @brief Add right-hand side vector to the Ruiz scaling handler.
         *  @param rhs2[in] - Pointer to the bottom right-hand side vector.
         */
        void addRhs2(real_type* rhs2);

        /**
         *  @brief Get the scaling vector.
         *  @return Pointer to the scaling vector.
         */
        real_type* getAggregateScalingVector() const;

        /**
         *  @brief Compute the Ruiz scaling.
         */
        void scale(memory::MemorySpace memspace);

      private:
        RuizScalingHandler* handler_;

        index_type num_iterations_;
        index_type n_;
        index_type total_n_;

        real_type* hes_v_;
        index_type* hes_i_;
        index_type* hes_j_;
        real_type* jac_v_;
        index_type* jac_i_;
        index_type* jac_j_;
        real_type* jac_tr_v_;
        index_type* jac_tr_i_;
        index_type* jac_tr_j_;
        real_type* rhs1_;
        real_type* rhs2_;

        real_type* scaling_vector_;
        real_type* aggregate_scaling_vector_;
    };
  }
}