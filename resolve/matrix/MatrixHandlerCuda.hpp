#pragma once
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/MatrixHandlerImpl.hpp>

namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }

  namespace matrix
  {
    class Sparse;
    class Coo;
    class Csc;
    class Csr;
  } // namespace matrix
  class LinAlgWorkspaceCUDA;
} // namespace ReSolve

namespace ReSolve
{
  /**
   * @class MatrixHandlerCuda
   *
   * @brief CUDA implementation of the matrix handler.
   */
  class MatrixHandlerCuda : public MatrixHandlerImpl
  {
    using vector_type = vector::Vector;

  public:
    MatrixHandlerCuda(LinAlgWorkspaceCUDA* workspace);
    virtual ~MatrixHandlerCuda();

    int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr) override;

    int transpose(matrix::Csr* A, matrix::Csr* At) override;

    int addConst(matrix::Sparse* A, real_type alpha) override;

    int leftScale(vector_type* diag, matrix::Csr* A) override;

    int rightScale(matrix::Csr* A, vector_type* diag) override;

    virtual int matvec(matrix::Sparse*  A,
                       vector_type*     vec_x,
                       vector_type*     vec_result,
                       const real_type* alpha,
                       const real_type* beta) override;
    virtual int matrixInfNorm(matrix::Sparse* A, real_type* norm) override;

    void setValuesChanged(bool isValuesChanged) override;

  private:
    LinAlgWorkspaceCUDA* workspace_{nullptr};
    bool                 values_changed_{true}; ///< Flag to indicate if matrix values changed since the cached SpMV setup.

    // The handler does not own or copy this matrix. It only tracks which matrix
    // structure was used to build the cached backend SpMV setup so the setup can
    // be reset when a different matrix is passed to matvec().
    matrix::Sparse* matrix_for_matvec_{nullptr}; ///< Matrix used for cached SpMV setup.
    index_type      matvec_num_rows_{0};         ///< Number of rows in cached SpMV matrix.
    index_type      matvec_num_cols_{0};         ///< Number of columns in cached SpMV matrix.
    index_type      matvec_nnz_{0};              ///< Number of nonzeros in cached SpMV matrix.

    MemoryHandler mem_; ///< Device memory manager object
  };

} // namespace ReSolve
