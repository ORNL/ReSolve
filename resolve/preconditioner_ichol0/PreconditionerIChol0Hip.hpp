#pragma once

#include "PreconditionerIChol0Impl.hpp"

#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <vector>

namespace ReSolve
{
  class PreconditionerIChol0Hip : public PreconditionerIChol0Impl
  {
  public:
    PreconditionerIChol0Hip(LinAlgWorkspaceHIP* workspace);
    ~PreconditionerIChol0Hip() override;

    int setup(matrix::Csr* L) override;
    int apply(vector::Vector* rhs, vector::Vector* x) override;
    void setNumRhs(index_type num_rhs) override;

  private:
    void freeData();
    int  analysis();

    LinAlgWorkspaceHIP* workspace_{nullptr};
    matrix::Csr*        L_{nullptr};

    rocsparse_mat_descr L_descr_{nullptr};

    std::vector<rocsparse_mat_info> L_info_;
    std::vector<rocsparse_mat_info> L_tr_info_;

    void*  L_buffer_{nullptr};
    size_t L_buffer_size_{0};

    void*  L_tr_buffer_{nullptr};
    size_t L_tr_buffer_size_{0};

    std::vector<hipStream_t> streams_;
    hipEvent_t               start_event_{nullptr};
    std::vector<hipEvent_t>  end_events_;

    index_type k_{1};
  };
} // namespace ReSolve