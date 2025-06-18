
#pragma once

#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <rocsparse/rocsparse.h>

#include "Common.hpp"
#include <resolve/LinSolverDirect.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  // Forward declaration of matrix::Sparse class
  namespace matrix
  {
    class Sparse;
  }

  class LinSolverDirectRocSparseILU0 : public LinSolverDirect
  {
    using vector_type = vector::Vector;

  public:
    LinSolverDirectRocSparseILU0(LinAlgWorkspaceHIP* workspace);
    ~LinSolverDirectRocSparseILU0();

    int setup(matrix::Sparse* A,
              matrix::Sparse* L   = nullptr,
              matrix::Sparse* U   = nullptr,
              index_type*     P   = nullptr,
              index_type*     Q   = nullptr,
              vector_type*    rhs = nullptr) override;
    // if values of A change, but the nnz pattern does not, redo the analysis only (reuse buffers though)
    int reset(matrix::Sparse* A);

    int solve(vector_type* rhs, vector_type* x) override;
    int solve(vector_type* rhs) override; // the solution is returned IN RHS (rhs is overwritten)

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    rocsparse_status status_rocsparse_;

    MemoryHandler       mem_; ///< Device memory manager object
    LinAlgWorkspaceHIP* workspace_{nullptr};

    rocsparse_mat_descr descr_A_{nullptr};
    rocsparse_mat_descr descr_L_{nullptr};
    rocsparse_mat_descr descr_U_{nullptr};

    rocsparse_mat_info info_A_{nullptr};

    void* buffer_{nullptr};

    real_type* d_aux1_{nullptr};
    // since ILU OVERWRITES THE MATRIX values, we need a buffer to keep the values of ILU decomposition.
    real_type* d_ILU_vals_{nullptr};
  };
} // namespace ReSolve
