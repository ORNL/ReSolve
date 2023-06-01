#pragma once
#include "LinAlgWorkspace.hpp"
#include "Vector.hpp"

namespace ReSolve
{
  class VectorHandler { 
    public:
      VectorHandler();
      VectorHandler(LinAlgWorkspace* new_workspace);
      ~VectorHandler();

      //y = alpha x + y
      void axpy( Real* alpha, Vector* x, Vector* y, std::string memspace );

      //dot: x \cdot y
      Real dot(Vector* x, Vector* y, std::string memspace );

      //scal = alpha * x
      void scal(Real* alpha, Vector* x, std::string memspace);

      //mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
      void massAxpy(Int size, Real* alpha, Real k, Real* x, Real* y,  std::string memspace);

      //mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
      //Size = n
      Real* massDot2Vec(Int size, Real* V, Real k, Real* x, Real* res,  std::string memspace);

      //gemv:
      //if transpose = N(no), x = beta*x +  alpha*V*y,
      //where x is [n x 1], V is [n x k] and y is [k x 1]
      //if transpose =T(yes), x = beta*x + alpha*V^T*y
      //where x is [k x 1], V is [n x k] and y is [n x 1] 
      void gemv(std::string transpose, Int n, Int k, Real* alpha, Real* beta, Real* V, Real* y, Real* x, std::string memspace);
    private:
      LinAlgWorkspace* workspace_;
      Real one_ = 1.0;
      Real minusone_ = -1.0;
      Real zero_ = 0.0;
  };

}
