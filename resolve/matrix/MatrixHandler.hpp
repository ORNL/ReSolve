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
  class LinAlgWorkspaceCpu;
  class LinAlgWorkspaceCUDA;
  class MatrixHandlerImpl;
}


namespace ReSolve {

  /**
   * @brief this class encapsulates various matrix manipulation operations, 
   * commonly required by linear solvers. 
   * 
   * This includes:
   *  - Matrix format conversion: coo2csr, csr2csc
   *  - Matrix vector product (SpMV)
   *  - Matrix 1-norm
   * 
   * @author Kasia Swirydowicz <kasia.swirydowicz@pnnl.gov>
   * @author Slaven Peles <peless@ornl.gov>
   */
  class MatrixHandler
  {
    using vector_type = vector::Vector;
    
    public:
      MatrixHandler();
      MatrixHandler(LinAlgWorkspaceCpu* workspace);
      MatrixHandler(LinAlgWorkspaceCUDA* workspace);
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
      void setValuesChanged(bool toWhat, std::string memspace); 
    
    private: 
      bool new_matrix_{true};     ///< if the structure changed, you need a new handler.

      MemoryHandler mem_; ///< Device memory manager object
      MatrixHandlerImpl*  cpuImpl_{nullptr};
      MatrixHandlerImpl* cudaImpl_{nullptr};

      bool isCpuEnabled_{false};
      bool isCudaEnabled_{false};
  };

} // namespace ReSolve

