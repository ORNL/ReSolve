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
  class LinAlgWorkspace;
  class MatrixHandlerImpl;
}


namespace ReSolve {

  //helper class
  class indexPlusValue
  {
    public:
      indexPlusValue();
      ~indexPlusValue();
      void setIdx (index_type new_idx);
      void setValue (real_type new_value);

      index_type getIdx();
      real_type getValue();

      bool operator < (const indexPlusValue& str) const
      {
        return (idx_ < str.idx_);
      }  

    private:
      index_type idx_;
      real_type value_;
  };

  class MatrixHandler
  {
    using vector_type = vector::Vector;
    
    public:
      MatrixHandler();
      MatrixHandler(LinAlgWorkspace* workspace);
      ~MatrixHandler();

      int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr, std::string memspace); //memspace decides on what is returned (cpu or cuda pointer)
      int coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, std::string memspace);

      /// Should compute vec_result := alpha*A*vec_x + beta*vec_result, but at least on cpu alpha and beta are flipped
      int matvec(matrix::Sparse* A,
                 vector_type* vec_x,
                 vector_type* vec_result,
                 const real_type* alpha,
                 const real_type* beta,
                 std::string matrix_type,
                 std::string memspace);
      int Matrix1Norm(matrix::Sparse *A, real_type* norm);
      void setValuesChanged(bool toWhat); 
    
    private: 
      LinAlgWorkspace* workspace_{nullptr};
      bool new_matrix_{true};     ///< if the structure changed, you need a new handler.
      bool values_changed_{true}; ///< needed for matvec

      MemoryHandler mem_; ///< Device memory manager object
      MatrixHandlerImpl* cpuImpl_{nullptr};
      MatrixHandlerImpl* cudaImpl_{nullptr};
  };

} // namespace ReSolve

