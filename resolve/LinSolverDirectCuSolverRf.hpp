#pragma once

#include "Common.hpp"
#include "cusolverRf.h"
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
    class Csr;
    class Csc;
  } // namespace matrix

  // Forward declaration of ReSolve handlers workspace
  class LinAlgWorkspaceCUDA;

  class LinSolverDirectCuSolverRf : public LinSolverDirect
  {
    using vector_type = vector::Vector;

  public:
    LinSolverDirectCuSolverRf(LinAlgWorkspaceCUDA* workspace = nullptr);
    ~LinSolverDirectCuSolverRf();

    int setup(matrix::Sparse* A,
              matrix::Sparse* L,
              matrix::Sparse* U,
              index_type*     P,
              index_type*     Q,
              vector_type*    rhs = nullptr) override;

    int refactorize() override;
    int solve(vector_type* rhs, vector_type* x) override;
    int solve(vector_type* rhs) override; // rhs overwritten by solution

    void setAlgorithms(cusolverRfFactorization_t   fact_alg,
                       cusolverRfTriangularSolve_t solve_alg);
    int  setNumericalProperties(real_type nzero, real_type nboost);

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    void initParamList();
    int  csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr);

  private:
    enum ParamaterIDs
    {
      ZERO_PIVOT = 0,
      PIVOT_BOOST
    };

    real_type zero_pivot_{0.0};  ///< The value below which zero pivot is flagged.
    real_type pivot_boost_{0.0}; ///< The value which is substituted for zero pivot.

    cusolverRfHandle_t handle_cusolverrf_;
    cusolverStatus_t   status_cusolverrf_;

    index_type* d_P_{nullptr};
    index_type* d_Q_{nullptr};
    real_type*  d_T_{nullptr};
    bool        setup_completed_{false};

    MemoryHandler mem_; ///< Device memory manager object
  };
} // namespace ReSolve
