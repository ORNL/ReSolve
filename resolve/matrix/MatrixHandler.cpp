#include <algorithm>
#include <cassert>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include "MatrixHandler.hpp"
#include "MatrixHandlerCpu.hpp"

#ifdef RESOLVE_USE_CUDA
#include "MatrixHandlerCuda.hpp"
#endif
#ifdef RESOLVE_USE_HIP
#include "MatrixHandlerHip.hpp"
#endif

namespace ReSolve {
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  /**
   * @brief Default constructor
   *
   * @post Instantiates CPU and CUDA matrix handlers, but does not
   * create a workspace.
   *
   * @todo There is little utility for the default constructor. Rethink its purpose.
   * Consider making it private method.
   */
  MatrixHandler::MatrixHandler()
  {
    new_matrix_ = true;
    cpuImpl_    = new MatrixHandlerCpu();
  }

  /**
   * @brief Destructor
   *
   */
  MatrixHandler::~MatrixHandler()
  {
    delete cpuImpl_;
    if (isCudaEnabled_ || isHipEnabled_) {
      delete devImpl_;
    }
  }

  /**
   * @brief Constructor taking pointer to the workspace as its parameter.
   *
   * @note The CPU implementation currently does not require a workspace.
   * The workspace pointer parameter is provided for forward compatibility.
   */
  MatrixHandler::MatrixHandler(LinAlgWorkspaceCpu* new_workspace)
  {
    cpuImpl_  = new MatrixHandlerCpu(new_workspace);
    isCpuEnabled_  = true;
    isCudaEnabled_ = false;
  }

#ifdef RESOLVE_USE_CUDA
  /**
   * @brief Constructor taking pointer to the CUDA workspace as its parameter.
   *
   * @post A CPU implementation instance is created because it is cheap and
   * it does not require a workspace.
   *
   * @post A CUDA implementation instance is created with supplied workspace.
   */
  MatrixHandler::MatrixHandler(LinAlgWorkspaceCUDA* new_workspace)
  {
    cpuImpl_ = new MatrixHandlerCpu();
    devImpl_ = new MatrixHandlerCuda(new_workspace);
    isCpuEnabled_  = true;
    isCudaEnabled_ = true;
  }
#endif

#ifdef RESOLVE_USE_HIP
  /**
   * @brief Constructor taking pointer to the CUDA workspace as its parameter.
   *
   * @post A CPU implementation instance is created because it is cheap and
   * it does not require a workspace.
   *
   * @post A HIP implementation instance is created with supplied workspace.
   */
  MatrixHandler::MatrixHandler(LinAlgWorkspaceHIP* new_workspace)
  {
    cpuImpl_ = new MatrixHandlerCpu();
    devImpl_ = new MatrixHandlerHip(new_workspace);
    isCpuEnabled_ = true;
    isHipEnabled_ = true;
  }
#endif
  /**
   * @brief Set the flag indicating that the matrix values have changed.
   *
   * If set to true, next invocation of `matvec` method will trigger re-allocation of the
   * matrix descriptor object. Use if the matrix changes or if matrix object internal pointers
   * to the raw matrix data change. This method has no effect if you are using CPU backend.
   *
   * @warning This is an expert-level method. Use only if you know what you are doing.
   *
   * @param[in] isValuesChanged - true if the values have changed, false otherwise
   * @param[in] memspace - Memory where values have changed
   */
  void MatrixHandler::setValuesChanged(bool isValuesChanged, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpuImpl_->setValuesChanged(isValuesChanged);
        break;
      case DEVICE:
        devImpl_->setValuesChanged(isValuesChanged);
        break;
    }
  }

  /**
   * @brief Matrix vector product: result = alpha * A * x + beta * result
   *
   * @param[in]  A - Sparse matrix
   * @param[in]  vec_x - Vector multiplied by the matrix
   * @param[out] vec_result - Vector where the result is stored
   * @param[in]  alpha - scalar parameter
   * @param[in]  beta  - scalar parameter
   * @param[in]  memspace     - Device where the product is computed
   * @param[in,out] result := alpha * A * x + beta * result
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandler::matvec(matrix::Sparse* A,
                            vector_type* vec_x,
                            vector_type* vec_result,
                            const real_type* alpha,
                            const real_type* beta,
                            memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        return cpuImpl_->matvec(A, vec_x, vec_result, alpha, beta);
        break;
      case DEVICE:
        return devImpl_->matvec(A, vec_x, vec_result, alpha, beta);
        break;
    }
    return 1;
  }

  /**
   * @bried Matrix inifinity norm (maximum absolute row sum)
   *
   * @param[in]  A - Sparse matrix
   * @param[out] norm - Maximum absolute row sum
   * @param[in]  memspace - Device where the norm is computed
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandler::matrixInfNorm(matrix::Sparse *A, real_type* norm, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        return cpuImpl_->matrixInfNorm(A, norm);
        break;
      case DEVICE:
        return devImpl_->matrixInfNorm(A, norm);
        break;
    }
    return 1;
  }

  /**
   * @brief Converts a CSC matrix to a CSR matrix
   *
   * @param[in]  A_csc - CSC matrix
   * @param[out] A_csr - CSR matrix
   * @param[in]  memspace - Device where the conversion is computed
   *
   * @pre `A_csc` is allocated, prefilled and has the correct CSC format.
   * @pre `A_csr` is allocated (but not prefilled) and has the correct CSR format.
   * @pre `A_csc` and `A_csr` are of the same size and have the same number of
   * nonzeros.
   *
   * @post `A_csr` is filled with the values from `A_csc`
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandler::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        return cpuImpl_->csc2csr(A_csc, A_csr);
        break;
      case DEVICE:
        return devImpl_->csc2csr(A_csc, A_csr);
        break;
    }
    return 1;
  }

  /**
   * @brief Transpose a sparse CSR matrix.
   *
   * @param[in]  A - Sparse matrix
   * @param[out] At - Transposed matrix
   * @param[in]  memspace - Device where the transpose is computed
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandler::transpose(matrix::Csr* A, matrix::Csr* At, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    // check dimensions of A and At and if both are allocated
    assert(A->getNumRows() == At->getNumColumns() && "Number of rows in A must be equal to number of columns in At");
    assert(A->getNumColumns() == At->getNumRows() && "Number of columns in A must be equal to number of rows in At");
    assert(A->getNnz() == At->getNnz() && "Number of nonzeros in A must be equal to number of nonzeros in At");
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
          "Matrix has to be in CSR format for transpose.\n");
    assert(At->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
          "Matrix has to be in CSR format for transpose.\n");
    switch (memspace) {
      case HOST:
        if(A->getValues(memory::HOST) == nullptr) {
          out::error() << "In MatrixHandler::transpose, A->getValues(memory::HOST) is null!\n";
          return 1;
        }
        if(At->getValues(memory::HOST) == nullptr) {
          out::error() << "In MatrixHandler::transpose, At->getValues(memory::HOST) is null!\n";
          return 1;
        }
        return cpuImpl_->transpose(A, At);
        break;
      case DEVICE:
        if(A->getValues(memory::DEVICE) == nullptr) {
          out::error() << "In MatrixHandlerCuda::transpose, A->getValues(memory::DEVICE) is null!\n";
          return 1;
        }
        if(At->getValues(memory::DEVICE) == nullptr) {
          out::error() << "In MatrixHandlerCuda::transpose, At->getValues(memory::DEVICE) is null!\n";
          return 1;
        }
        return devImpl_->transpose(A, At);
        break;
    }
    return 1;
  }

  /**
   * @brief Left diagonal scaling of a sparse CSR matrix
   *
   * @param[in]  A - Sparse CSR matrix
   * @param[in]  diag - vector representing the diagonal matrix
   * @param[in]  memspace - Device where the operation is computed
   *
   * @pre The diagonal vector must be of the same size as the number of rows in the matrix.
   * @pre A is unscaled and allocated
   * @post A is scaled
   * @invariant diag
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandler::leftDiagonalScale(vector_type* diag, matrix::Csr* A, memory::MemorySpace memspace)
  {
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
           "Matrix has to be in CSR format for left diagonal scaling.\n");
    assert(diag->getSize() == A->getNumRows() && 
           "Diagonal vector must be of the same size as the number of rows in the matrix.");
    assert(A->getValues(memspace) != nullptr && "Matrix values are null!\n");
    assert(diag->getData(memspace) != nullptr && "Diagonal vector data is null!\n");
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        return cpuImpl_->leftDiagonalScale(diag, A);
        break;
      case DEVICE:
        return devImpl_->leftDiagonalScale(diag, A);
        break;
    }
    return 1;
  }

  /**
   * @brief Right diagonal scaling of a sparse CSR matrix
   *
   * @param[in]  A - Sparse CSR matrix
   * @param[in]  diag - vector representing the diagonal matrix
   * @param[in]  memspace - Device where the operation is computed
   *
   * @pre The diagonal vector must be of the same size as the number of columns in the matrix.
   * @pre A is unscaled and allocated
   * @post A is scaled
   * @invariant diag
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandler::rightDiagonalScale(matrix::Csr* A, vector_type* diag, memory::MemorySpace memspace)
  {
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW &&
           "Matrix has to be in CSR format for right diagonal scaling.\n");
    assert(diag->getSize() == A->getNumColumns() && "Diagonal vector must be of the same size as the number of columns in the matrix.");
    assert(A->getValues(memspace) != nullptr && "Matrix values are null!\n");
    assert(diag->getData(memspace) != nullptr && "Diagonal vector data is null!\n");
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        return cpuImpl_->rightDiagonalScale(A, diag);
        break;
      case DEVICE:
        return devImpl_->rightDiagonalScale(A, diag);
        break;
    }
    return 1;
  }

  /**
   * @brief Add a constant to the nonzero values of a csr matrix.
   * @param[in,out] A - Sparse matrix
   * @param[in] alpha - scalar parameter
   * @param[in] memspace - Device where the operation is computed
   * @return 0 if successful, 1 otherwise
   */
  void MatrixHandler::addConst(matrix::Sparse* A, real_type alpha, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpuImpl_->addConst(A, alpha);
        break;
      case DEVICE:
        devImpl_->addConst(A, alpha);
        break;
    }
  }

  /**
   * @brief If CUDA support is enabled in the handler.
   *
   * @return true
   * @return false
   */
  bool MatrixHandler::getIsCudaEnabled() const
  {
    return isCudaEnabled_;
  }

  /**
   * @brief If HIP support is enabled in the handler.
   *
   * @return true
   * @return false
   */
  bool MatrixHandler::getIsHipEnabled() const
  {
    return isHipEnabled_;
  }

} // namespace ReSolve
