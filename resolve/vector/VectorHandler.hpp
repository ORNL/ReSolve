#pragma once
#include <resolve/LinAlgWorkspace.hpp>

namespace ReSolve { namespace vector {
  class Vector;
}}

namespace ReSolve { //namespace vector {
  class VectorHandler { 
    public:
      VectorHandler();
      VectorHandler(LinAlgWorkspace* new_workspace);
      ~VectorHandler();

      //y = alpha x + y
      void axpy( real_type* alpha, vector::Vector* x, vector::Vector* y, std::string memspace );

      //dot: x \cdot y
      real_type dot(vector::Vector* x, vector::Vector* y, std::string memspace );

      //scal = alpha * x
      void scal(real_type* alpha, vector::Vector* x, std::string memspace);

      //mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
      void massAxpy(index_type size, real_type* alpha, real_type k, real_type* x, real_type* y,  std::string memspace);

      //mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
      //Size = n
      void massDot2Vec(index_type size, real_type* V, real_type k, real_type* x, real_type* res,  std::string memspace);

      //gemv:
      //if transpose = N(no), x = beta*x +  alpha*V*y,
      //where x is [n x 1], V is [n x k] and y is [k x 1]
      //if transpose =T(yes), x = beta*x + alpha*V^T*y
      //where x is [k x 1], V is [n x k] and y is [n x 1] 
      void gemv(std::string transpose, index_type n, index_type k, real_type* alpha, real_type* beta, real_type* V, real_type* y, real_type* x, std::string memspace);
    private:
      LinAlgWorkspace* workspace_;
      real_type one_ = 1.0;
      real_type minusone_ = -1.0;
      real_type zero_ = 0.0;
  };

} //} // namespace ReSolve::vector
