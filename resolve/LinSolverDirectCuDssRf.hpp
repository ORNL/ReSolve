#pragma once

#include <cudss.h>

#include "Common.hpp"
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

  class LinSolverDirectCuDssRf : public LinSolverDirect
  {
    using vector_type = vector::Vector;

  public:
    LinSolverDirectCuDssRf(LinAlgWorkspaceCUDA* workspace = nullptr);
    ~LinSolverDirectCuDssRf();

    // For backward compatibility.
    int setup(matrix::Sparse* A,
              matrix::Sparse*,
              matrix::Sparse*,
              index_type* P,
              index_type* Q,
              vector_type* = nullptr) override;

    int setup(matrix::Sparse* A,
              index_type*     P);

    int refactorize() override;
    int solve(vector_type* rhs, vector_type* x) override;
    int solve(vector_type* rhs) override; // rhs overwritten by solution

    int setNumericalProperties(real_type nzero, real_type); // For backward compatibility
    int setNumericalProperties(real_type nzero);

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

    real_type zero_pivot_{0.0}; ///< The value below which zero pivot is flagged.

    cudssHandle_t handle_cudss_{nullptr};
    cudssConfig_t config_cudss_{nullptr};
    cudssData_t   data_cudss_{nullptr};
    cudssMatrix_t descr_A_{nullptr};

    index_type* d_P_{nullptr};
    bool        setup_completed_{false};

    MemoryHandler mem_; ///< Device memory manager object
  };
} // namespace ReSolve
