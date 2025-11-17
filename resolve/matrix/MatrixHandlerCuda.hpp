#pragma once

#include <cusparse.h>

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

    int scaleAddI(matrix::Csr* A, real_type alpha) override;

    int scaleAddB(matrix::Csr* A, real_type alpha, matrix::Csr* B) override;

    virtual int matvec(matrix::Sparse*  A,
                       vector_type*     vec_x,
                       vector_type*     vec_result,
                       const real_type* alpha,
                       const real_type* beta) override;
    virtual int matrixInfNorm(matrix::Sparse* A, real_type* norm) override;

    void setValuesChanged(bool isValuesChanged) override;

  private:
    void                 allocateForSum(matrix::Csr* A, real_type alpha, matrix::Csr* B, real_type beta, matrix::Csr* C, cusparseMatDescr_t& descr_a, void** buffer_add);
    void                 compute_sum(matrix::Csr* A, real_type alpha, matrix::Csr* B, real_type beta, matrix::Csr* C, cusparseMatDescr_t& descr_a, void** buffer_add);
    LinAlgWorkspaceCUDA* workspace_{nullptr};
    bool                 values_changed_{true}; ///< needed for matvec

    MemoryHandler mem_; ///< Device memory manager object
  };

} // namespace ReSolve
