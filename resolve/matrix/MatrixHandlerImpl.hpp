// this class encapsulates various matrix manipulation operations, commonly required by linear solvers:
// this includes 
// (1) Matrix format conversion: coo2csr, csr2csc
// (2) Matrix vector product (SpMV)
// (3) Matrix 1-norm
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
  // class LinAlgWorkspace;
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
      // MatrixHandlerImpl(LinAlgWorkspace* workspace);
      virtual ~MatrixHandlerImpl()
      {}

      virtual int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr) = 0;
      // int coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, std::string memspace);

      /// Should compute vec_result := alpha*A*vec_x + beta*vec_result, but at least on cpu alpha and beta are flipped
      virtual int matvec(matrix::Sparse* A,
                         vector_type* vec_x,
                         vector_type* vec_result,
                         const real_type* alpha,
                         const real_type* beta,
                         std::string matrix_type) = 0;
      virtual int Matrix1Norm(matrix::Sparse* A, real_type* norm) = 0;

      virtual void setValuesChanged(bool isValuesChanged) = 0;
    
    protected: 
      // LinAlgWorkspace* workspace_{nullptr};
      // bool new_matrix_{true};     ///< if the structure changed, you need a new handler.
      // bool values_changed_{true}; ///< needed for matvec

      // MemoryHandler mem_; ///< Device memory manager object
  };

} // namespace ReSolve

