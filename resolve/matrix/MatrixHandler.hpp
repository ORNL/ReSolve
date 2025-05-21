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
  class LinAlgWorkspaceHIP;
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
   *  - Matrix Inf-norm
   *
   * The class uses pointer to implementation (PIMPL) idiom to create
   * multiple matrix operation implementations running on CUDA and HIP devices
   * as well as on CPU.
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
      MatrixHandler(LinAlgWorkspaceHIP* workspace);
      ~MatrixHandler();

      int csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr, memory::MemorySpace memspace);

      int transpose(matrix::Csr* A, matrix::Csr* At, memory::MemorySpace memspace);

      int leftDiagonalScale(vector_type* diag, matrix::Csr* A, memory::MemorySpace memspace);

      int rightDiagonalScale(matrix::Csr* A, vector_type* diag, memory::MemorySpace memspace);

      void addConst(matrix::Sparse* A, real_type alpha, memory::MemorySpace memspace);



      /// Should compute vec_result := alpha*A*vec_x + beta*vec_result, but at least on cpu alpha and beta are flipped
      int matvec(matrix::Sparse* A,
                 vector_type* vec_x,
                 vector_type* vec_result,
                 const real_type* alpha,
                 const real_type* beta,
                 memory::MemorySpace memspace);
      int matrixInfNorm(matrix::Sparse *A, real_type* norm, memory::MemorySpace memspace);
      void setValuesChanged(bool toWhat, memory::MemorySpace memspace);

      bool getIsCudaEnabled() const;
      bool getIsHipEnabled()  const;

    private:
      bool new_matrix_{true};  ///< if the structure changed, you need a new handler.

      MatrixHandlerImpl* cpuImpl_{nullptr}; ///< Pointer to host implementation
      MatrixHandlerImpl* devImpl_{nullptr}; ///< Pointer to device implementation

      bool isCpuEnabled_{false};  ///< true if CPU  implementation is instantiated
      bool isCudaEnabled_{false}; ///< true if CUDA implementation is instantiated
      bool isHipEnabled_{false};  ///< true if HIP  implementation is instantiated
  };

} // namespace ReSolve

