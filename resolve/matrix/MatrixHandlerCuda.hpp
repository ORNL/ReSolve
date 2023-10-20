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
  class LinAlgWorkspaceCUDA;
}


namespace ReSolve {
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

      int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr);
      virtual int matvec(matrix::Sparse* A,
                 vector_type* vec_x,
                 vector_type* vec_result,
                 const real_type* alpha,
                 const real_type* beta,
                 std::string matrix_type);
      virtual int Matrix1Norm(matrix::Sparse *A, real_type* norm);
      void setValuesChanged(bool isValuesChanged); 
    
    private: 
      LinAlgWorkspaceCUDA* workspace_{nullptr};
      bool values_changed_{true}; ///< needed for matvec

      MemoryHandler mem_; ///< Device memory manager object
  };

} // namespace ReSolve

