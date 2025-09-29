#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

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

  // Forward declaration of matrix::Sparse and matrix::Csr classes
  namespace matrix
  {
    class Sparse;
    class Csr;
  } // namespace matrix

  class LinSolverDirectRocSolverRf : public LinSolverDirect
  {
    using vector_type = vector::Vector;

  public:
    LinSolverDirectRocSolverRf(LinAlgWorkspaceHIP* workspace);
    ~LinSolverDirectRocSolverRf();

    int setup(matrix::Sparse* A,
              matrix::Sparse* L,
              matrix::Sparse* U,
              index_type*     P,
              index_type*     Q,
              vector_type*    rhs) override;

    int refactorize() override;
    int solve(vector_type* rhs, vector_type* x) override;
    int solve(vector_type* rhs) override; // the solution overwrites rhs

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    enum ParamaterIDs
    {
    };

  private:
    // to be exported to matrix handler in a later time
    void combineFactors(matrix::Sparse* L, matrix::Sparse* U); // create L+U from separate L, U factors
    void initParamList();

    rocblas_status   status_rocblas_;
    index_type*      d_P_{nullptr};
    index_type*      d_Q_{nullptr};

    MemoryHandler       mem_; ///< Device memory manager object
    LinAlgWorkspaceHIP* workspace_;

    rocsolver_rfinfo infoM_;
    matrix::Sparse*  M_{nullptr}; // the matrix that contains added factors

    ReSolve::matrix::Csr* L_csr_{nullptr};
    ReSolve::matrix::Csr* U_csr_{nullptr};

    real_type* d_aux1_{nullptr};
    real_type* d_aux2_{nullptr};
  };
} // namespace ReSolve
