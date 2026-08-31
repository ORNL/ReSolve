/**
 * @file   PreconditionerIChol0.hpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Declaration of preconditioner LU class.
 *
 */

#pragma once

#include "PreconditionerIChol0Impl.hpp"
#ifdef RESOLVE_USE_CUDA
#include "PreconditionerIChol0Cuda.hpp"
#elif defined(RESOLVE_USE_HIP)
#include "PreconditionerIChol0Hip.hpp"
#endif

#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/Preconditioner.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{
  class PreconditionerIChol0 : public Preconditioner
  {
  public:
    PreconditionerIChol0(MatrixHandler* matrix_handler, LinAlgWorkspaceCpu* workspace);
#ifdef RESOLVE_USE_CUDA
    PreconditionerIChol0(MatrixHandler* matrix_handler, LinAlgWorkspaceCUDA* workspace);
#elif defined(RESOLVE_USE_HIP)
    PreconditionerIChol0(MatrixHandler* matrix_handler, LinAlgWorkspaceHIP* workspace);
#endif
    ~PreconditionerIChol0();

    int setup(matrix_type* A) override;
    int apply(vector_type* rhs, vector_type* x) override;
    int reset(matrix_type* A) override;
    void setNumRhs(index_type num_rhs);
    void setNumericBoost(real_type numeric_boost);

  private:
    matrix::Csr* L_{nullptr};
    real_type numeric_boost_ = 0;
    MatrixHandler* matrix_handler_{nullptr};

    PreconditionerIChol0Impl* impl_{nullptr};
  };
} // namespace ReSolve