#pragma once
#include <string>

namespace ReSolve
{ 
  namespace vector
  {
    class Vector;
  }
  class VectorHandlerCpu;
  class VectorHandlerCuda;
}


namespace ReSolve
{
  class VectorHandlerImpl
  { 
    public:
      VectorHandlerImpl()
      {}
      virtual ~VectorHandlerImpl()
      {}

      //y = alpha x + y
      virtual void axpy(const real_type* alpha, vector::Vector* x, vector::Vector* y ) = 0;

      //dot: x \cdot y
      virtual real_type dot(vector::Vector* x, vector::Vector* y ) = 0;

      //scal = alpha * x
      virtual void scal(const real_type* alpha, vector::Vector* x) = 0;

      //mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
      virtual void massAxpy(index_type size, vector::Vector* alpha, index_type k, vector::Vector* x, vector::Vector* y) = 0;

      //mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
      //Size = n
      virtual void massDot2Vec(index_type size, vector::Vector* V, index_type k, vector::Vector* x, vector::Vector* res) = 0;

      /** gemv:
       * if `transpose = N` (no), `x = beta*x +  alpha*V*y`,
       * where `x` is `[n x 1]`, `V` is `[n x k]` and `y` is `[k x 1]`.
       * if `transpose = T` (yes), `x = beta*x + alpha*V^T*y`,
       * where `x` is `[k x 1]`, `V` is `[n x k]` and `y` is `[n x 1]`.
       */ 
      virtual void gemv(std::string transpose,
                        index_type n,
                        index_type k,
                        const real_type* alpha,
                        const real_type* beta,
                        vector::Vector* V,
                        vector::Vector* y,
                        vector::Vector* x) = 0;
  };

} //} // namespace ReSolve::vector
