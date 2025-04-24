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
  }
  class LinAlgWorkspaceHIP;
}


namespace ReSolve {
  /**
   * @class MatrixHandlerHip
   *
   * @brief HIP implementation of the matrix handler.
   */
  class MatrixHandlerHip : public MatrixHandlerImpl
  {
    using vector_type = vector::Vector;

    public:

      MatrixHandlerHip(LinAlgWorkspaceHIP* workspace);
      virtual ~MatrixHandlerHip();

      int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr);

      int transpose(matrix::Csr* A, matrix::Csr* At) override;

      int addConst(matrix::Sparse* A, real_type alpha);
      virtual int matvec(matrix::Sparse* A,
                         vector_type* vec_x,
                         vector_type* vec_result,
                         const real_type* alpha,
                         const real_type* beta);

      virtual int matrixInfNorm(matrix::Sparse *A, real_type* norm);

      void setValuesChanged(bool isValuesChanged);

    private:

      LinAlgWorkspaceHIP* workspace_{nullptr};
      bool values_changed_{true}; ///< needed for matvec

      MemoryHandler mem_; ///< Device memory manager object
  };

} // namespace ReSolve

