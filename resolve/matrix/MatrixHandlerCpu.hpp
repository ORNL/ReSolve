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
  class LinAlgWorkspaceCpu;
} // namespace ReSolve

namespace ReSolve
{
  /**
   * @class MatrixHandlerCpu
   *
   * @brief CPU implementation of the matrix handler.
   */
  class MatrixHandlerCpu : public MatrixHandlerImpl
  {
    using vector_type = vector::Vector;

  public:
    MatrixHandlerCpu();
    MatrixHandlerCpu(LinAlgWorkspaceCpu* workspace);
    virtual ~MatrixHandlerCpu();

    int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr) override;

    int transpose(matrix::Csr* A, matrix::Csr* At) override;

    int leftScale(vector_type* diag, matrix::Csr* A) override;

    int rightScale(matrix::Csr* A, vector_type* diag) override;

    int addConst(matrix::Sparse* A, real_type alpha) override;

    virtual int matvec(matrix::Sparse*  A,
                       vector_type*     vec_x,
                       vector_type*     vec_result,
                       const real_type* alpha,
                       const real_type* beta) override;

    virtual int matrixInfNorm(matrix::Sparse* A, real_type* norm) override;

    void setValuesChanged(bool isValuesChanged) override;

  private:
    LinAlgWorkspaceCpu* workspace_{nullptr};
    bool                values_changed_{true}; ///< needed for matvec

    // MemoryHandler mem_; ///< Device memory manager object not used for now
  };

} // namespace ReSolve
