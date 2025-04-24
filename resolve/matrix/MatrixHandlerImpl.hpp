#pragma once

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
}


namespace ReSolve {
  /**
   * @class MatrixHandlerImpl
   *
   * @brief Base class for different matrix handler implementations.
   */
  class MatrixHandlerImpl
  {
    using vector_type = vector::Vector;

    public:
      MatrixHandlerImpl()
      {}
      virtual ~MatrixHandlerImpl()
      {}

      virtual int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr) = 0;

      virtual int transpose(matrix::Csr* A, matrix::Csr* At) = 0;

      virtual int addConst(matrix::Sparse* A, real_type alpha) = 0;

      virtual int matvec(matrix::Sparse* A,
                         vector_type* vec_x,
                         vector_type* vec_result,
                         const real_type* alpha,
                         const real_type* beta) = 0;
      virtual int matrixInfNorm(matrix::Sparse* A, real_type* norm) = 0;

      virtual void setValuesChanged(bool isValuesChanged) = 0;
  };

} // namespace ReSolve

