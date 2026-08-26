#pragma once

#include "PreconditionerIChol0Impl.hpp"

#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/MemoryUtils.hpp>

#include <cuda_runtime.h>
#include <cusparse.h>

namespace ReSolve
{
  class PreconditionerIChol0Cuda : public PreconditionerIChol0Impl
  {
  public:
    PreconditionerIChol0Cuda(LinAlgWorkspaceCUDA* workspace);
    ~PreconditionerIChol0Cuda() override;

    int setup(matrix::Csr* L) override;
    int apply(vector::Vector* rhs, vector::Vector* x) override;
    void setNumRhs(index_type num_rhs) override;

  private:
    void freeData();
    int  analysis();

    index_type k_{1};
    matrix::Csr*         L_{nullptr};

    cusparseMatDescr_t L_descr_{nullptr};
    cusparseSpMatDescr_t mat_L_{nullptr};
    csric02Info_t      info_{nullptr};

    cusparseDnVecDescr_t vec_b_{nullptr};
    cusparseDnVecDescr_t vec_x_{nullptr};
    cusparseSpSVDescr_t  L_descr_spsv_{nullptr};
    cusparseSpSVDescr_t  L_tr_descr_spsv_{nullptr};
    void*   L_buffer_spsv_{nullptr};
    void*   L_tr_buffer_spsv_{nullptr};
    size_t  L_buffer_size_spsv_{0};
    size_t  L_tr_buffer_size_spsv_{0};

    // Variables for SpSM (k > 1)
    cusparseDnMatDescr_t mat_B_{nullptr};
    cusparseDnMatDescr_t mat_X_{nullptr};
    void*  L_buffer_spsm_{nullptr};
    void*  L_tr_buffer_spsm_{nullptr};
    size_t L_buffer_size_spsm_{0};
    size_t L_tr_buffer_size_spsm_{0};

    cusparseSpSMDescr_t L_descr_spsm_{nullptr};
    cusparseSpSMDescr_t L_tr_descr_spsm_{nullptr};

    LinAlgWorkspaceCUDA* workspace_{nullptr};
    MemoryHandler mem_;
  };
} // namespace ReSolve