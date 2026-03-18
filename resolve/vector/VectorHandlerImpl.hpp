#pragma once

namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }
  class VectorHandlerCpu;
  class VectorHandlerCuda;
} // namespace ReSolve

namespace ReSolve
{
  class VectorHandlerImpl
  {
  public:
    VectorHandlerImpl()
    {
    }

    virtual ~VectorHandlerImpl()
    {
    }

    // y = alpha x + y
    virtual void axpy(const real_type alpha, vector::Vector* x, vector::Vector* y) = 0;

    // dot: x \cdot y
    virtual real_type dot(vector::Vector* x, vector::Vector* y) = 0;

    // scal = alpha * x
    virtual void scal(const real_type alpha, vector::Vector* x) = 0;

    // amax = ||x||_\infty
    virtual real_type amax(vector::Vector* x) = 0;

    // mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
    virtual void axpyMulti(index_type size, vector::Vector* alpha, index_type k, vector::Vector* x, vector::Vector* y) = 0;

    // mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
    // Size = n
    virtual void dot2Multi(index_type size, vector::Vector* V, index_type k, vector::Vector* x, vector::Vector* res) = 0;

    // Scale a vector by a diagonal matrix
    virtual void scal(vector::Vector* diag, vector::Vector* vec) = 0;

    // Divide the elements of a vector by the elements of another vector
    virtual int diagSolve(vector::Vector* diag, vector::Vector* vec) = 0;

    // Compute element-wise max of two vectors
    virtual int max(/* const */ vector::Vector* x, /* const */ vector::Vector* y, vector::Vector* out) = 0;

    // Compute element-wise absolute value of a vector
    virtual int abs(/* const */ vector::Vector* in, vector::Vector* out) = 0;

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
                      vector::Vector* x) = 0;
  };

} // namespace ReSolve
