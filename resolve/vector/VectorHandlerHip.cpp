#include "VectorHandlerHip.hpp"

#include <cassert>
#include <iostream>

#include <resolve/hip/hipKernels.h>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandlerImpl.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief empty constructor that does absolutely nothing
   */
  VectorHandlerHip::VectorHandlerHip()
  {
  }

  /**
   * @brief constructor
   *
   * @param new_workspace - workspace to be set
   */
  VectorHandlerHip::VectorHandlerHip(LinAlgWorkspaceHIP* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /**
   * @brief destructor
   */
  VectorHandlerHip::~VectorHandlerHip()
  {
    // delete the workspace TODO
  }

  /**
   * @brief dot product of two vectors i.e, a = x^Ty
   *
   * @param[in] x The first vector
   * @param[in] y The second vector
   *
   * @return dot product (real number) of _x_ and _y_
   */

  real_type VectorHandlerHip::dot(vector::Vector* x, vector::Vector* y)
  {
    rocblas_handle handle_rocblas = workspace_->getRocblasHandle();
    double         nrm            = 0.0;

    rocblas_status st = rocblas_ddot(handle_rocblas,
                                     x->getSize(),
                                     x->getData(memory::DEVICE),
                                     1,
                                     y->getData(memory::DEVICE),
                                     1,
                                     &nrm);
    if (st != 0)
    {
      out::error() << "dot product returned error code " << st << "\n";
    }
    return nrm;
  }

  /**
   * @brief scale a vector by a constant i.e, x = alpha*x where alpha is a constant
   *
   * @param[in] alpha The constant
   * @param[in,out] x The vector
   *
   */
  void VectorHandlerHip::scal(const real_type alpha, vector::Vector* x)
  {
    rocblas_handle handle_rocblas = workspace_->getRocblasHandle();

    rocblas_status st = rocblas_dscal(handle_rocblas,
                                      x->getSize(),
                                      &alpha,
                                      x->getData(memory::DEVICE),
                                      1);
    if (st != 0)
    {
      out::error() << "scal returned error code " << st << "\n";
    }
    x->setDataUpdated(memory::DEVICE);
  }

  /**
   * @brief compute infinity norm of a vector (i.e., find an entry with largest absolute value)
   *
   * @param[in] The vector
   *
   * @return infinity norm (real number) of _x_
   *
   */
  real_type VectorHandlerHip::amax(vector::Vector* x)
  {

    if (workspace_->getNormBufferState() == false)
    { // not allocated
      real_type* buffer;
      mem_.allocateArrayOnDevice(&buffer, 1024);
      workspace_->setNormBuffer(buffer);
      workspace_->setNormBufferState(true);
    }
    real_type norm{0.0};
    hip::vector_inf_norm(x->getSize(),
                         x->getData(memory::DEVICE),
                         workspace_->getNormBuffer(),
                         &norm);
    return norm;
  }

  /**
   * @brief axpy i.e, y = alpha*x + y where alpha is a constant
   *
   * @param[in] alpha The constant
   * @param[in] x The first vector
   * @param[in,out] y The second vector (result is return in y)
   *
   */
  void VectorHandlerHip::axpy(const real_type alpha, vector::Vector* x, vector::Vector* y)
  {
    rocblas_handle handle_rocblas = workspace_->getRocblasHandle();
    rocblas_daxpy(handle_rocblas,
                  x->getSize(),
                  &alpha,
                  x->getData(memory::DEVICE),
                  1,
                  y->getData(memory::DEVICE),
                  1);
    y->setDataUpdated(memory::DEVICE);
  }

  /**
   * @brief gemv computes matrix-vector product where both matrix and vectors are dense.
   *        i.e., x = beta*x +  alpha*V*y
   *
   * @param[in] Transpose - yes (T) or no (N)
   * @param[in] n Number of rows in (non-transposed) matrix
   * @param[in] k Number of columns in (non-transposed)
   * @param[in] alpha Constant real number
   * @param[in] beta Constant real number
   * @param[in] V Multivector containing the matrix, organized columnwise
   * @param[in] y Vector, k x 1 if N and n x 1 if T
   * @param[in,out] x Vector, n x 1 if N and k x 1 if T
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
  void VectorHandlerHip::gemv(char            transpose,
                              index_type      k,
                              const real_type alpha,
                              const real_type beta,
                              vector::Vector* V,
                              vector::Vector* y,
                              vector::Vector* x)
  {
    rocblas_handle   handle_rocblas = workspace_->getRocblasHandle();
    const index_type n              = V->getSize();

    switch (transpose)
    {
    case 'T':
      assert((V->getSize() == y->getSize())
             && "gemv: Size mismatch! Size of V does not match size of y.");
      rocblas_dgemv(handle_rocblas,
                    rocblas_operation_transpose,
                    n,
                    k,
                    &alpha,
                    V->getData(memory::DEVICE),
                    n,
                    y->getData(memory::DEVICE),
                    1,
                    &beta,
                    x->getData(memory::DEVICE),
                    1);
      return;
    default:
      assert((V->getSize() == x->getSize())
             && "gemv: Size mismatch! Size of V does not match size of x.");
      rocblas_dgemv(handle_rocblas,
                    rocblas_operation_none,
                    n,
                    k,
                    &alpha,
                    V->getData(memory::DEVICE),
                    n,
                    y->getData(memory::DEVICE),
                    1,
                    &beta,
                    x->getData(memory::DEVICE),
                    1);
      if (transpose != 'N')
      {
        out::warning() << "Unrecognized transpose option " << transpose
                       << " in gemv. Using non-transposed multivector.\n";
      }
    }
    x->setDataUpdated(memory::DEVICE);
  }

  /**
   * @brief mass (bulk) axpy i.e, y = y - x*alpha where  alpha is a vector
   *
   * @param[in] size number of elements in y
   * @param[in] alpha vector size k x 1
   * @param[in] x (multi)vector size size x k
   * @param[in,out] y vector size size x 1 (this is where the result is stored)
   *
   * @pre   _k_ > 0, _size_ > 0, _size_ = x->getSize()
   *
   */
  void VectorHandlerHip::axpyMulti(index_type      size,
                                   vector::Vector* alpha,
                                   index_type      k,
                                   vector::Vector* x,
                                   vector::Vector* y)
  {
    using namespace constants;
    if (k < 200)
    {
      hip::axpy_multi(size,
                      k,
                      x->getData(memory::DEVICE),
                      y->getData(memory::DEVICE),
                      alpha->getData(memory::DEVICE));
    }
    else
    {
      rocblas_handle handle_rocblas = workspace_->getRocblasHandle();
      rocblas_dgemm(handle_rocblas,
                    rocblas_operation_none,
                    rocblas_operation_none,
                    size,                           // m
                    1,                              // n
                    k,                              // k
                    &MINUS_ONE,                     // alpha
                    x->getData(memory::DEVICE),     // A
                    size,                           // lda
                    alpha->getData(memory::DEVICE), // B
                    k,                              // ldb
                    &ONE,
                    y->getData(memory::DEVICE), // c
                    size);                      // ldc
    }
    y->setDataUpdated(memory::DEVICE);
  }

  /**
   * @brief mass (bulk) dot product i.e,  V^T x, where V is n x k dense multivector
   * (a dense multivector consisting of k vectors size n) and x is k x 2 dense
   * multivector (a multivector consisiting of two vectors size n each)
   *
   * @param[in] size Number of elements in a single vector in V
   * @param[in] V Multivector; k vectors size n x 1 each
   * @param[in] k Number of vectors in V
   * @param[in] x Multivector; 2 vectors size n x 1 each
   * @param[out] res Multivector; 2 vectors size k x 1 each (result is returned in res)
   *
   * @pre   _size_ > 0, _k_ > 0, size = x->getSize(), _res_ needs to be allocated
   *
   */
  void VectorHandlerHip::dot2Multi(index_type      size,
                                   vector::Vector* V,
                                   index_type      k,
                                   vector::Vector* x,
                                   vector::Vector* res)
  {
    using namespace constants;

    if (k < 200)
    {
      hip::dot_2_multi(size,
                       k,
                       x->getData(0, memory::DEVICE),
                       x->getData(1, memory::DEVICE),
                       V->getData(memory::DEVICE),
                       res->getData(memory::DEVICE));
    }
    else
    {
      rocblas_handle handle_rocblas = workspace_->getRocblasHandle();
      rocblas_dgemm(handle_rocblas,
                    rocblas_operation_transpose,
                    rocblas_operation_none,
                    k,                          // m
                    2,                          // n
                    size,                       // k
                    &ONE,                       // alpha
                    V->getData(memory::DEVICE), // A
                    size,                       // lda
                    x->getData(memory::DEVICE), // B
                    size,                       // ldb
                    &ZERO,
                    res->getData(memory::DEVICE), // c
                    k);                           // ldc
    }
    res->setDataUpdated(memory::DEVICE);
  }

  /**
   * @brief Scale a vector by a diagonal matrix in HIP
   *
   * @param[in]  diag - vector representing the diagonal matrix
   * @param[in, out]  vec - vector to be scaled
   *
   * @pre The diagonal vector must be of the same size as the vector.
   * @pre vec is unscaled
   * @post vec is scaled
   * @invariant diag
   *
   * @return 0 if successful, 1 otherwise
   */
  void VectorHandlerHip::scal(vector::Vector* diag, vector::Vector* vec)
  {
    real_type* diag_data = diag->getData(memory::DEVICE);
    real_type* vec_data  = vec->getData(memory::DEVICE);
    index_type n         = vec->getSize();
    hip::scale(n, diag_data, vec_data);
    vec->setDataUpdated(memory::DEVICE);
  }

  /**
   * @brief Multiplies vector by an inverse of a diagonal matrix.
   *
   * @param[in]  diag   - diagonal matrix stored in a vector object
   * @param[in, out]  vec - vector to be divided
   *
   * @pre The diagonal vector must be of the same size as the vector.
   * @pre vec is undivided
   * @post vec is divided
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandlerHip::diagSolve(vector::Vector* diag, vector::Vector* vec)
  {
    real_type* diag_data = diag->getData(memory::DEVICE);
    real_type* vec_data  = vec->getData(memory::DEVICE);
    index_type n         = vec->getSize();
    hip::diagSolve(n, diag_data, vec_data);
    vec->setDataUpdated(memory::DEVICE);
    return 0;
  }

  /**
   * @brief Calculate element-wise maximum between two vectors in HIP
   *
   * @param[in]  x   - The first vector
   * @param[in]  y   - The second vector
   * @param[out] out - The output vector
   *
   * @pre The three vectors must be the same size
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandlerHip::max(/* const */ vector::Vector* x, /* const */ vector::Vector* y, vector::Vector* out)
  {
    real_type* x_data   = x->getData(memory::DEVICE);
    real_type* y_data   = y->getData(memory::DEVICE);
    real_type* out_data = out->getData(memory::DEVICE);
    index_type n        = y->getSize();
    hip::max(n, x_data, y_data, out_data);
    out->setDataUpdated(memory::DEVICE);
    return 0;
  }

  /**
   * @brief Calculate element-wise absolute value of a vector in HIP
   *
   * @param[in]  in  - The input vector
   * @param[out] out - The output
   *
   * @return 0 if successful, 1 otherwise
   */
  int VectorHandlerHip::abs(/* const */ vector::Vector* in, vector::Vector* out)
  {
    const real_type* in_data  = in->getData(memory::DEVICE);
    real_type*       out_data = out->getData(memory::DEVICE);
    index_type       n        = in->getSize();
    hip::abs(n, in_data, out_data);
    out->setDataUpdated(memory::DEVICE);
    return 0;
  }

} // namespace ReSolve
