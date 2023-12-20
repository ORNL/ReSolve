#include <iostream>
#include <cmath>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/vector/VectorHandlerImpl.hpp>
#include <resolve/vector/VectorHandlerCpu.hpp>
#include "VectorHandler.hpp"

#ifdef RESOLVE_USE_CUDA
#include <resolve/vector/VectorHandlerCuda.hpp>
#endif
#ifdef RESOLVE_USE_HIP
#include <resolve/vector/VectorHandlerHip.hpp>
#endif

namespace ReSolve {
  using out = io::Logger;

  /** 
   * @brief empty constructor that does absolutely nothing        
   */
  VectorHandler::VectorHandler()
  {
    cpuImpl_ = new VectorHandlerCpu();
    isCpuEnabled_ = true;
  }

  /** 
   * @brief constructor
   * 
   * @param new_workspace - workspace to be set     
   */
  VectorHandler::VectorHandler(LinAlgWorkspaceCpu* new_workspace)
  {
    cpuImpl_ = new VectorHandlerCpu(new_workspace);
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
    cpuImpl_  = new  VectorHandlerCpu();

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
    devImpl_  = new VectorHandlerHip(new_workspace);
    cpuImpl_  = new VectorHandlerCpu();

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
    if (isCudaEnabled_ || isHipEnabled_) {
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
    switch (memspace) {
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
  void VectorHandler::scal(const real_type* alpha, vector::Vector* x, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpuImpl_->scal(alpha, x);
        break;
      case DEVICE:
        devImpl_->scal(alpha, x);
        break;
    }
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
  real_type VectorHandler::infNorm(vector::Vector* x, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        return cpuImpl_->infNorm(x);
        break;
      case DEVICE:
        return devImpl_->infNorm(x);
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
  void VectorHandler::axpy(const  real_type* alpha, vector::Vector* x, vector::Vector* y, memory::MemorySpace memspace)
  {
    //AXPY:  y = alpha * x + y
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpuImpl_->axpy(alpha, x, y);
        break;
      case DEVICE:
        devImpl_->axpy(alpha, x, y);      
        break;
    }
  }

  /** 
   * @brief gemv computes matrix-vector product where both matrix and vectors are dense.
   *        i.e., x = beta*x +  alpha*V*y
   *
   * @param[in] Transpose - yes (T) or no (N)
   * @param[in] n         - Number of rows in (non-transposed) matrix
   * @param[in] k         - Number of columns in (non-transposed)   
   * @param[in] alpha     - Constant real number
   * @param[in] beta      - Constant real number
   * @param[in] V         - Multivector containing the matrix, organized columnwise
   * @param[in] y         - Vector, k x 1 if N and n x 1 if T
   * @param[in,out] x     - Vector, n x 1 if N and k x 1 if T
   * @param[in] memspace  - cpu or cuda or hip (for now)
   *
   * @pre   V is stored colum-wise, _n_ > 0, _k_ > 0
   * 
   */  
  void VectorHandler::gemv(char transpose,
                           index_type n,
                           index_type k,
                           const real_type* alpha,
                           const real_type* beta,
                           vector::Vector* V,
                           vector::Vector* y,
                           vector::Vector* x,
                           memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpuImpl_->gemv(transpose, n, k, alpha, beta, V, y, x);
        break;
      case DEVICE:
        devImpl_->gemv(transpose, n, k, alpha, beta, V, y, x);
        break;
    }
  }

  /** 
   * @brief mass (bulk) axpy i.e, y = y - x*alpha where  alpha is a vector
   * 
   * @param[in] size number of elements in y
   * @param[in] alpha vector size k x 1
   * @param[in] x (multi)vector size size x k
   * @param[in,out] y vector size size x 1 (this is where the result is stored)
   * @param[in] memspace string containg memspace (cpu or cuda or hip)
   *
   * @pre   _k_ > 0, _size_ > 0, _size_ = x->getSize()
   *
   */
  void VectorHandler::massAxpy(index_type size, vector::Vector* alpha, index_type k, vector::Vector* x, vector::Vector* y, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpuImpl_->massAxpy(size, alpha, k, x, y);
        break;
      case DEVICE:
        devImpl_->massAxpy(size, alpha, k, x, y);
        break;
    }
  }

  /** 
   * @brief mass (bulk) dot product i.e,  V^T x, where V is n x k dense multivector (a dense multivector consisting of k vectors size n)  
   *        and x is k x 2 dense multivector (a multivector consisiting of two vectors size n each)
   * 
   * @param[in] size     - Number of elements in a single vector in V
   * @param[in] V        - Multivector; k vectors size n x 1 each
   * @param[in] k        - Number of vectors in V
   * @param[in] x        - Multivector; 2 vectors size n x 1 each
   * @param[out] res     - Multivector; 2 vectors size k x 1 each (result is returned in res)
   * @param[in] memspace - String containg memspace (cpu or cuda or hip)
   *
   * @pre   _size_ > 0, _k_ > 0, size = x->getSize(), _res_ needs to be allocated
   *
   */
  void VectorHandler::massDot2Vec(index_type size, vector::Vector* V, index_type k, vector::Vector* x, vector::Vector* res, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpuImpl_->massDot2Vec(size, V, k, x, res);
        break;
      case DEVICE:
        devImpl_->massDot2Vec(size, V, k, x, res);
        break;
    }
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
