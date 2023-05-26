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
      void axpy( Real* alpha, Vector* x, Vector* y );

      //dot: x \cdot y
      Real dot(Vector* x, Vector* y, std::string memspace );

      //scal = alpha * x
      void scal(Real* alpha, Vector* x);

      //mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
      void mass_axpy(Int size, Real* alpha, Real k, Real* x,Real* y);

      //mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
      Real* mass_dot(Real size, Real* V, Real k, Real* x);

    private:
      LinAlgWorkspace* workspace_;
  };

}
