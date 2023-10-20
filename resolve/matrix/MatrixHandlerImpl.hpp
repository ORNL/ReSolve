#pragma once
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

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

      virtual int matvec(matrix::Sparse* A,
                         vector_type* vec_x,
                         vector_type* vec_result,
                         const real_type* alpha,
                         const real_type* beta,
                         std::string matrix_type) = 0;
      virtual int Matrix1Norm(matrix::Sparse* A, real_type* norm) = 0;

      virtual void setValuesChanged(bool isValuesChanged) = 0;    
  };

} // namespace ReSolve

