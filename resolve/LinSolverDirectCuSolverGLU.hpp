#pragma once

#include "Common.hpp"
#include "cusolver_defs.hpp"
#include <resolve/LinSolverDirect.hpp>
#include <resolve/MemoryUtils.hpp>

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

  // Forward declaration of ReSolve handlers workspace
  class LinAlgWorkspaceCUDA;

  class LinSolverDirectCuSolverGLU : public LinSolverDirect
  {
    using vector_type = vector::Vector;

  public:
    LinSolverDirectCuSolverGLU(LinAlgWorkspaceCUDA* workspace);
    ~LinSolverDirectCuSolverGLU();

    int refactorize() override;
    int solve(vector_type* rhs, vector_type* x) override;
    int solve(vector_type* x) override;

    int setup(matrix::Sparse* A,
              matrix::Sparse* L,
              matrix::Sparse* U,
              index_type*     P,
              index_type*     Q,
              vector_type*    rhs = nullptr) override;

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    void            combineFactors(matrix::Sparse* L, matrix::Sparse* U); ///< creates L+U from separate L, U factors
    matrix::Sparse* M_;                                                   ///< the matrix that contains added factors
    // note: we need cuSolver handle, we can copy it from the workspace to avoid double allocation
    cusparseMatDescr_t   descr_M_;   // this is NOT sparse matrix descriptor
    cusparseMatDescr_t   descr_A_;   // this is NOT sparse matrix descriptor
    LinAlgWorkspaceCUDA* workspace_; ///< Workspace access so we can copy cusparse handle
    cusolverSpHandle_t   handle_cusolversp_;
    cusolverStatus_t     status_cusolver_;
    cusparseStatus_t     status_cusparse_;
    csrgluInfo_t         info_M_;
    void*                glu_buffer_;
    double               r_nrm_inf_;       ///< Error norm of the solution
    int                  ite_refine_succ_; ///< Stores return value of cusolverSpDgluSolve

    MemoryHandler mem_; ///< Device memory manager object
  };
} // namespace ReSolve
