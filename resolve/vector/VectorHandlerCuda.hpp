#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/VectorHandlerImpl.hpp>

namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }
  class LinAlgWorkspaceCUDA;
} // namespace ReSolve

namespace ReSolve
{ // namespace vector {

  class VectorHandlerCuda : public VectorHandlerImpl
  {
  public:
    VectorHandlerCuda();
    VectorHandlerCuda(LinAlgWorkspaceCUDA* workspace);
    virtual ~VectorHandlerCuda();

    // y = alpha x + y
    virtual void axpy(const real_type alpha, /* const */ vector::Vector* x, vector::Vector* y);

    // dot: x \cdot y
    virtual real_type dot(vector::Vector* x, vector::Vector* y);

    // scal = alpha * x
    virtual void scal(const real_type alpha, vector::Vector* x);

    // vector infinity norm
    virtual real_type amax(vector::Vector* x);

    // mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
    virtual void axpyMulti(index_type size, vector::Vector* alpha, index_type k, vector::Vector* x, vector::Vector* y);

    // mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
    // Size = n
    virtual void dot2Multi(index_type size, vector::Vector* V, index_type k, vector::Vector* x, vector::Vector* res);

    /** gemv:
     * if `transpose = N` (no), `x = beta*x +  alpha*V*y`,
     * where `x` is `[n x 1]`, `V` is `[n x k]` and `y` is `[k x 1]`.
     * if `transpose = T` (yes), `x = beta*x + alpha*V^T*y`,
     * where `x` is `[k x 1]`, `V` is `[n x k]` and `y` is `[n x 1]`.
     */
    virtual void gemv(char            transpose,
                      index_type      k,
                      const real_type alpha,
                      const real_type beta,
                      vector::Vector* V,
                      vector::Vector* y,
                      vector::Vector* x);

    /**
     * @brief scale: scales a vector by a diagonal matrix
     *
     * @param[in] diag diagonal vector of size n x 1
     * @param[in,out] vec vector of size n x 1 (this is where the result is stored)
     *
     * @return 0 if successful, 1 otherwise
     */
    virtual void scal(vector::Vector* diag, vector::Vector* vec);

    /**
     * @brief scale: scales a vector by a diagonal matrix represented by a contiguous subvector of an input vector
     *
     * @param[in] diag diagonal vector of size n x 1
     * @param[in,out] vec vector of size n x 1 (this is where the result is stored)
     * @param[in] diag_offset - the index of diag where the diagonal matrix begins offset
     *
     * @return 0 if successful, 1 otherwise
     */
    virtual void scal(vector::Vector* diag, vector::Vector* vec, index_type diag_offset);

    /**
     * @brief Multiplies vector by an inverse of a diagonal matrix.
     *
     *
     * @param[in] diag vector of size n x 1
     * @param[in,out] vec vector of size n x 1 (this is where the result is stored)
     *
     * @return 0 if successful, 1 otherwise
     */
    virtual int diagSolve(vector::Vector* diag, vector::Vector* vec);

    /**
     * @brief max: calculate the element-wise maximum of two vectors
     *
     * @param[in]  x   vector of size n x 1
     * @param[in]  y   vector of size n x 1
     * @param[out] out output vector of size n x 1
     *
     * @return 0 if successful, 1 otherwise
     */
    virtual int max(/* const */ vector::Vector* x, /* const */ vector::Vector* y, vector::Vector* out);

    /**
     * @brief abs: calculate the element-wise absolute value of a vector
     *
     * @param[in,out] x vector of size n x 1 (this is where the result is stored)
     *
     * @return 0 if successful, 1 otherwise
     */
    virtual int abs(/* const */ vector::Vector* in, vector::Vector* out);

  private:
    MemoryHandler        mem_; ///< Device memory manager object
    LinAlgWorkspaceCUDA* workspace_;
  };

} // namespace ReSolve
