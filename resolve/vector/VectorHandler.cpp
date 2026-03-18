#include "VectorHandler.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandlerCpu.hpp>
#include <resolve/vector/VectorHandlerImpl.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/vector/VectorHandlerCuda.hpp>
#endif
#ifdef RESOLVE_USE_HIP
#include <resolve/vector/VectorHandlerHip.hpp>
#endif

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief empty constructor that does absolutely nothing
   */
  VectorHandler::VectorHandler()
  {
    cpuImpl_      = new VectorHandlerCpu();
    isCpuEnabled_ = true;
  }

  /**
   * @brief constructor
   *
   * @param new_workspace - workspace to be set
   */
  VectorHandler::VectorHandler(LinAlgWorkspaceCpu* new_workspace)
  {
    cpuImpl_      = new VectorHandlerCpu(new_workspace);
    isCpuEnabled_ = true;
  }

#ifdef RESOLVE_USE_CUDA
  /**
   * @brief constructor
   *
   * @param new_workspace - workspace to be set
   */
  VectorHandler::VectorHandler(LinAlgWorkspaceCUDA* new_workspace)
  {
    devImpl_ = new VectorHandlerCuda(new_workspace);
    cpuImpl_ = new VectorHandlerCpu();

    isCudaEnabled_ = true;
    isCpuEnabled_  = true;
  }
#endif
#ifdef RESOLVE_USE_HIP
  /**
   * @brief constructor
   *
   * @param new_workspace - workspace to be set
   */
  VectorHandler::VectorHandler(LinAlgWorkspaceHIP* new_workspace)
  {
    devImpl_ = new VectorHandlerHip(new_workspace);
    cpuImpl_ = new VectorHandlerCpu();

    isHipEnabled_ = true;
    isCpuEnabled_ = true;
  }
#endif

  /**
   * @brief destructor
   */
  VectorHandler::~VectorHandler()
  {
    delete cpuImpl_;
    if (isCudaEnabled_ || isHipEnabled_)
    {
      delete devImpl_;
    }
  }

  /**
   * @brief dot product of two vectors i.e, a = x^Ty
   *
   * @param[in] x The first vector
   * @param[in] y The second vector
   * @param[in] memspace String containg memspace (cpu or cuda or hip)
   *
   * @return dot product (real number) of _x_ and _y_
   */

  real_type VectorHandler::dot(vector::Vector* x, vector::Vector* y, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      return cpuImpl_->dot(x, y);
      break;
    case DEVICE:
      return devImpl_->dot(x, y);
      break;
    }
    return NAN;
  }

  /**
   * @brief scale a vector by a constant i.e, x = alpha*x where alpha is a constant
   *
   * @param[in] alpha The constant
   * @param[in,out] x The vector
   * @param memspace[in] string containg memspace (cpu or cuda or hip)
   *
   */
  void VectorHandler::scal(const real_type alpha, vector::Vector* x, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      cpuImpl_->scal(alpha, x);
      break;
    case DEVICE:
      devImpl_->scal(alpha, x);
      break;
    }
    x->setDataUpdated(memspace);
  }

  /**
   * @brief compute infinity norm of a vector (i.e., find an entry with largest absolute value)
   *
   * @param[in] The vector
   * @param[in] memspace string containg memspace (cpu or cuda or hip)
   *
   * @return infinity norm (real number) of _x_
   *
   */
  real_type VectorHandler::amax(vector::Vector* x, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      return cpuImpl_->amax(x);
      break;
    case DEVICE:
      return devImpl_->amax(x);
      break;
    }
    return -1.0;
  }

  /**
   * @brief axpy i.e, y = alpha*x+y where alpha is a constant
   *
   * @param[in] alpha The constant
   * @param[in] x The first vector
   * @param[in,out] y The second vector (result is return in y)
   * @param[in]  memspace String containg memspace (cpu or cuda or hip)
   *
   */
  void VectorHandler::axpy(const real_type             alpha,
                           /* const */ vector::Vector* x,
                           vector::Vector*             y,
                           memory::MemorySpace         memspace)
  {
    // AXPY:  y = alpha * x + y
    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      cpuImpl_->axpy(alpha, x, y);
      break;
    case DEVICE:
      devImpl_->axpy(alpha, x, y);
      break;
    }
    y->setDataUpdated(memspace);
  }

  /**
   * @brief gemv computes dense matrix-vector product.
   *
   * In Re::Solve applications, gemv is used to compute dot products of
   * multivectors.
   *
   * If `transpose = N` (no), `x := beta*x +  alpha*V*y`,
   * where `x` is `[n x 1]`, `V` is `[n x k]` and `y` is `[k x 1]`.
   * If `transpose = T` (yes), `x := beta*x + alpha*V^T*y`,
   * where `x` is `[k x 1]`, `V` is `[n x k]` and `y` is `[n x 1]`.
   *
   * @param[in] Transpose - yes (T) or no (N)
   * @param[in] n         - Number of rows in (non-transposed) matrix
   * @param[in] k         - Number of columns in (non-transposed) matrix to use
   * @param[in] alpha     - Constant real number
   * @param[in] beta      - Constant real number
   * @param[in] V         - Multivector containing the matrix, organized columnwise
   * @param[in] y         - Vector, k x 1 if N and n x 1 if T
   * @param[in,out] x     - Vector, n x 1 if N and k x 1 if T
   * @param[in] memspace  - enum specifying HOST or DEVICE memory space.
   *
   * @note Parameter k is not the total number of columns in V but the number
   * of columns to use in matrix-vector product.
   *
   * @pre _n_ > 0, _k_ > 0
   * @pre Number of columns in V >= k
   * @pre If transpose = N, size of y must equal k. If transpose = T, size of
   * x must equal k.
   *
   */
  void VectorHandler::gemv(char                transpose,
                           index_type          k,
                           const real_type     alpha,
                           const real_type     beta,
                           vector::Vector*     V,
                           vector::Vector*     y,
                           vector::Vector*     x,
                           memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;

    switch (memspace)
    {
    case HOST:
      cpuImpl_->gemv(transpose, k, alpha, beta, V, y, x);
      break;
    case DEVICE:
      devImpl_->gemv(transpose, k, alpha, beta, V, y, x);
      break;
    }
    x->setDataUpdated(memspace);
  }

  /**
   * @brief Multivector axpy: y: = y + \sum_i alpha_i x_i
   *
   * @param[in] size number of elements in y
   * @param[in] alpha vector size k x 1
   * @param[in] x (multi)vector [size x k]
   * @param[in,out] y vector size size x 1 (this is where the result is stored)
   * @param[in] memspace string containg memspace (cpu, cuda, or hip)
   *
   * @pre   _k_ > 0, _size_ > 0, _size_ = x->getSize()
   *
   */
  void VectorHandler::axpyMulti(index_type          size,
                                vector::Vector*     alpha,
                                index_type          k,
                                vector::Vector*     x,
                                vector::Vector*     y,
                                memory::MemorySpace memspace)
  {
    assert(y->getSize() == x->getSize() && "Sizes of x and y must match!\n");
    assert(alpha->getSize() == k && "Size of alpha must match k!\n");

    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      cpuImpl_->axpyMulti(size, alpha, k, x, y);
      break;
    case DEVICE:
      devImpl_->axpyMulti(size, alpha, k, x, y);
      break;
    }
    y->setDataUpdated(memspace);
  }

  /**
   * @brief Multivector dot product, i.e  V^T x
   *
   * Computes V^T x with k vectors from multivector V. Result is stored
   * in `res`.
   *
   * @param[in] size     - Number of elements in a single vector in V
   * @param[in] V        - Multivector; k vectors of size n x 1 each
   * @param[in] k        - Number of vectors in V to use
   * @param[in] x        - Multivector; 2 vectors of size n x 1 each
   * @param[out] res     - Multivector; 2 vectors size k x 1 each
   * @param[in] memspace - String containg memspace (cpu, cuda, or hip)
   *
   * @pre _size_ > 0, _k_ > 0, size = x->getSize().
   * @pre _res_ needs to be allocated to k x 2 size.
   *
   */
  void VectorHandler::dot2Multi(index_type          size,
                                vector::Vector*     V,
                                index_type          k,
                                vector::Vector*     x,
                                vector::Vector*     res,
                                memory::MemorySpace memspace)
  {
    assert(x->getSize() == V->getSize() && "Sizes of V and x do not match!\n");
    assert(res->getSize() == k && "Size of `res` must match k!\n");

    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      cpuImpl_->dot2Multi(size, V, k, x, res);
      break;
    case DEVICE:
      devImpl_->dot2Multi(size, V, k, x, res);
      break;
    }
    res->setDataUpdated(memspace);
  }

  /**
   * @brief Scale a vector by a diagonal matrix
   *
   * @param[in] diag - vector representing the diagonal matrix
   * @param[in,out] vec - vector to be scaled
   * @param[in] memspace - Device where the operation is computed
   *
   * @pre The diagonal vector must be of the same size as the vector.
   * @invariant diag
   *
   */
  void VectorHandler::scal(vector::Vector* diag, vector::Vector* vec, memory::MemorySpace memspace)
  {
    assert(diag->getSize() == vec->getSize() && "Diagonal vector must be of the same size as the vector.");
    assert(diag->getData(memspace) != nullptr && "Diagonal vector data is null!\n");
    assert(vec->getData(memspace) != nullptr && "Vector data is null!\n");

    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      return cpuImpl_->scal(diag, vec);
      break;
    case DEVICE:
      return devImpl_->scal(diag, vec);
      break;
    }
  }

  /**
   * @brief Multiplies vector by an inverse of a diagonal matrix.
   *
   * @param[in]  diag   - diagonal matrix stored in a vector object
   * @param[in,out] vec - vector to be divided
   * @param[in] memspace - Device where the operation is computed
   *
   * @pre The two vectors must be the same size
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandler::diagSolve(vector::Vector* diag, vector::Vector* vec, memory::MemorySpace memspace)
  {
    assert(diag->getSize() == vec->getSize() && "Diagonal vector must be of the same size as the vector.");
    assert(diag->getData(memspace) != nullptr && "Diagonal vector data is null!\n");
    assert(vec->getData(memspace) != nullptr && "Vector data is null!\n");
    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      return cpuImpl_->diagSolve(diag, vec);
      break;
    case DEVICE:
      return devImpl_->diagSolve(diag, vec);
      break;
    }
    return 1;
  }

  /**
   * @brief Takes the element-wise max between two vectors.
   *
   * @param[in]  x        - The first vector
   * @param[in]  y        - The second vector
   * @param[out] out      - The output vector
   * @param[in]  memspace - Device where the operation is computed
   *
   * @pre The two vectors must be the same size
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandler::max(/* const */ vector::Vector* x, /* const */ vector::Vector* y, vector::Vector* out, memory::MemorySpace memspace)
  {
    assert(x->getSize() == y->getSize() && "Vectors must be the same size.");
    assert(x->getSize() == out->getSize() && "Vectors must be the same size.");
    assert(x->getData(memspace) != nullptr && "Vector x data is null!");
    assert(y->getData(memspace) != nullptr && "Vector y data is null!");
    assert(out->getData(memspace) != nullptr && "Vector out data is null!");
    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      return cpuImpl_->max(x, y, out);
      break;
    case DEVICE:
      return devImpl_->max(x, y, out);
      break;
    }
    return 1;
  }

  /**
   * @brief Computes the element-wise absolute value of a vector.
   *
   * @param[in]  in       - Input vector
   * @param[out] out      - Output vector
   * @param[in]  memspace - Device where the operation is computed
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandler::abs(/* const */ vector::Vector* in, vector::Vector* out, memory::MemorySpace memspace)
  {
    assert(in->getData(memspace) != nullptr && "Vector in data is null!");
    assert(out->getData(memspace) != nullptr && "Vector out data is null!");
    assert(in->getSize() == out->getSize() && "Vector sizes do not match!");

    using namespace ReSolve::memory;
    switch (memspace)
    {
    case HOST:
      return cpuImpl_->abs(in, out);
      break;
    case DEVICE:
      return devImpl_->abs(in, out);
      break;
    }
    return 1;
  }

  /**
   * @brief If CUDA support is enabled in the handler.
   *
   * @return true
   * @return false
   */
  bool VectorHandler::getIsCudaEnabled() const
  {
    return isCudaEnabled_;
  }

  /**
   * @brief If HIP support is enabled in the handler.
   *
   * @return true
   * @return false
   */
  bool VectorHandler::getIsHipEnabled() const
  {
    return isHipEnabled_;
  }

} // namespace ReSolve
