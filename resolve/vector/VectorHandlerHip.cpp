#include <iostream>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/hip/hipKernels.h>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/vector/VectorHandlerImpl.hpp>
#include "VectorHandlerHip.hpp"

namespace ReSolve {
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
  VectorHandlerHip:: VectorHandlerHip(LinAlgWorkspaceHIP* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /**
   * @brief destructor
   */
  VectorHandlerHip::~VectorHandlerHip()
  {
    //delete the workspace TODO
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
    double nrm = 0.0;
    rocblas_status st= rocblas_ddot (handle_rocblas,  x->getSize(), x->getData(memory::DEVICE), 1, y->getData(memory::DEVICE), 1, &nrm);

    if (st!=0) {printf("dot product crashed with code %d \n", st);}
    return nrm;
  }

  /**
   * @brief scale a vector by a constant i.e, x = alpha*x where alpha is a constant
   *
   * @param[in] alpha The constant
   * @param[in,out] x The vector
   *
   */
  void VectorHandlerHip::scal(const real_type* alpha, vector::Vector* x)
  {
    rocblas_handle handle_rocblas =  workspace_->getRocblasHandle();
    rocblas_status st = rocblas_dscal(handle_rocblas, x->getSize(), alpha, x->getData(memory::DEVICE), 1);

    if (st!=0) {
      ReSolve::io::Logger::error() << "scal crashed with code " << st << "\n";
    }
  }

  /**
   * @brief compute infinity norm of a vector (i.e., find an entry with largest absolute value)
   *
   * @param[in] The vector
   *
   * @return infinity norm (real number) of _x_
   *
   */
  real_type VectorHandlerHip::infNorm(vector::Vector* x)
  {

    if (workspace_->getNormBufferState() == false) { // not allocated
      real_type* buffer;
      mem_.allocateArrayOnDevice(&buffer, 1024);
      workspace_->setNormBuffer(buffer);
      workspace_->setNormBufferState(true);
    }
    real_type norm;
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
  void VectorHandlerHip::axpy(const  real_type* alpha, vector::Vector* x, vector::Vector* y)
  {
    rocblas_handle handle_rocblas =  workspace_->getRocblasHandle();
    rocblas_daxpy(handle_rocblas,
                  x->getSize(),
                  alpha,
                  x->getData(memory::DEVICE),
                  1,
                  y->getData(memory::DEVICE),
                  1);

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
   * @pre   V is stored colum-wise, _n_ > 0, _k_ > 0
   *
   */
  void VectorHandlerHip::gemv(char transpose,
                              index_type n,
                              index_type k,
                              const real_type* alpha,
                              const real_type* beta,
                              vector::Vector* V,
                              vector::Vector* y,
                              vector::Vector* x)
  {
    rocblas_handle handle_rocblas =  workspace_->getRocblasHandle();
    switch (transpose) {
      case 'T':
        rocblas_dgemv(handle_rocblas,
                      rocblas_operation_transpose,
                      n,
                      k,
                      alpha,
                      V->getData(memory::DEVICE),
                      n,
                      y->getData(memory::DEVICE),
                      1,
                      beta,
                      x->getData(memory::DEVICE),
                      1);
        return;
      default:
        rocblas_dgemv(handle_rocblas,
                      rocblas_operation_none,
                      n,
                      k,
                      alpha,
                      V->getData(memory::DEVICE),
                      n,
                      y->getData(memory::DEVICE),
                      1,
                      beta,
                      x->getData(memory::DEVICE),
                      1);
        if (transpose != 'N') {
          out::warning() << "Unrecognized transpose option " << transpose
                         << " in gemv. Using non-transposed multivector.\n";
        }
    }
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
  void VectorHandlerHip::massAxpy(index_type size, vector::Vector* alpha, index_type k, vector::Vector* x, vector::Vector* y)
  {
    using namespace constants;
    if (k < 200) {
      hip::mass_axpy(size, k, x->getData(memory::DEVICE), y->getData(memory::DEVICE),alpha->getData(memory::DEVICE));

    } else {
      rocblas_handle handle_rocblas =  workspace_->getRocblasHandle();
      rocblas_dgemm(handle_rocblas,
                    rocblas_operation_none,
                    rocblas_operation_none,
                    size,       // m
                    1,          // n
                    k,      // k
                    &MINUS_ONE, // alpha
                    x->getData(memory::DEVICE), // A
                    size,       // lda
                    alpha->getData(memory::DEVICE), // B
                    k,      // ldb
                    &ONE,
                    y->getData(memory::DEVICE),          // c
                    size);      // ldc

    }
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
  void VectorHandlerHip::massDot2Vec(index_type size, vector::Vector* V, index_type k, vector::Vector* x, vector::Vector* res)
  {
    using namespace constants;

    if (k < 200) {
      hip::mass_inner_product_two_vectors(size, k, x->getData(0, memory::DEVICE) , x->getData(1, memory::DEVICE), V->getData(memory::DEVICE), res->getData(memory::DEVICE));
    } else {
      rocblas_handle handle_rocblas =  workspace_->getRocblasHandle();
      rocblas_dgemm(handle_rocblas,
                    rocblas_operation_transpose,
                    rocblas_operation_none,
                    k,   //m
                    2,       //n
                    size,    //k
                    &ONE,   //alpha
                    V->getData(memory::DEVICE),       //A
                    size,    //lda
                    x->getData(memory::DEVICE),       //B
                    size,    //ldb
                    &ZERO,
                    res->getData(memory::DEVICE),     //c
                    k);  //ldc

    }
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
  int VectorHandlerHip::scale(vector::Vector* diag, vector::Vector* vec)
  {
    real_type* diag_data = diag->getData(memory::DEVICE);
    real_type* vec_data = vec->getData(memory::DEVICE);
    index_type n = vec->getSize();
    hip::scale(n, diag_data, vec_data);
    vec->setDataUpdated(memory::DEVICE);
    return 0;
  }

} // namespace ReSolve
