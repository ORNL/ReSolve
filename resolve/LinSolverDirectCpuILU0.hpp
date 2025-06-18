/**
 * @file LinSolverDirectCpuILU0.hpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Contains declaration of a class for incomplete LU factorization on CPU
 *
 *
 */
#pragma once

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
  }

  // Forward declaration of CPU workspace
  class LinAlgWorkspaceCpu;

  /**
   * @brief Incomplete LU factorization solver.
   *
   * Implements ILU0 factorization from Algorithm 1 in 2023 paper by Suzuki,
   * Fukaya, and Iwashita with modification where zero diagonal elements in
   * the matrix are replaced by small values specified in `zero_diagonal_`.
   * Factors L and U are stored in separate CSR matrices. Factor L does not
   * store ones at the diagonal.
   *
   * Methods in this class perform all operations on raw matrix data.
   *
   */
  class LinSolverDirectCpuILU0 : public LinSolverDirect
  {
    using vector_type = vector::Vector;

  public:
    LinSolverDirectCpuILU0(LinAlgWorkspaceCpu* workspace = nullptr);
    ~LinSolverDirectCpuILU0();

    int setup(matrix::Sparse* A,
              matrix::Sparse* L   = nullptr,
              matrix::Sparse* U   = nullptr,
              index_type*     P   = nullptr,
              index_type*     Q   = nullptr,
              vector_type*    rhs = nullptr) override;
    // if values of A change, but the nnz pattern does not, redo the analysis only (reuse buffers though)
    int reset(matrix::Sparse* A);
    int analyze() override;
    int factorize() override;

    int solve(vector_type* rhs, vector_type* x) override;
    int solve(vector_type* rhs) override; // the solution is returned IN RHS (rhs is overwritten)

    matrix::Sparse* getLFactor() override;
    matrix::Sparse* getUFactor() override;

    int setZeroDiagonal(real_type z);

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    // MemoryHandler mem_; ///< Device memory manager object
    // LinAlgWorkspaceCpu* workspace_{nullptr};

    matrix::Csr* A_{nullptr};          ///< Pointer to the system matrix
    real_type*   diagU_{nullptr};      ///< Buffer holding diagonal of factor U
    index_type*  idxmap_{nullptr};     ///< Mapping for matrix column indices
    bool         owns_factors_{false}; ///< If the class owns L and U factors

    real_type zero_diagonal_{1e-6}; ///< Approximation for zero diagonal
  };
} // namespace ReSolve
