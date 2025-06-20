namespace ReSolve {
  namespace hykkt {

    class RuizScalingKernelImpl {
    public:
      virtual void adaptRowMax(index_type n_hes, index_type nnz_hes, index_type* hes_i, index_type* hes_j, real_type* hes_v,
                         index_type n_jac, index_type m_jac, index_type* jac_i, index_type* jac_j, real_type* jac_v,
                         index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v,
                         real_type* rhs1, real_type* rhs2,
                         real_type* aggregate_scaling_vector,
                         real_type* scaling_vector) = 0;

      virtual void adaptDiagScale(index_type n_hes, index_type nnz_hes, index_type* hes_i, index_type* hes_j, real_type* hes_v,
                         index_type n_jac, index_type m_jac, index_type* jac_i, index_type* jac_j, real_type* jac_v,
                         index_type* jac_tr_i, index_type* jac_tr_j, real_type* jac_tr_v,
                         real_type* rhs1, real_type* rhs2,
                         real_type* aggregate_scaling_vector,
                         real_type* scaling_vector) = 0;
    };

  } // namespace hykkt
}