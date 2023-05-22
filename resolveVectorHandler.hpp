#pragma once
#include "resolveLinAlgWorkspace.hpp"
#include "resolveVector.hpp"

namespace ReSolve
{
  class resolveVectorHandler { 
    public:
      resolveVectorHandler();
      resolveVectorHandler(resolveLinAlgWorkspace* new_workspace);
      ~resolveVectorHandler();

      //y = alpha x + y
      void axpy( resolveReal* alpha, resolveVector* x, resolveVector* y );

      //dot: x \cdot y
      resolveReal dot(resolveVector* x, resolveVector* y, std::string memspace );

      //scal = alpha * x
      void scal(resolveReal* alpha, resolveVector* x);

      //mass axpy: x*alpha + y where x is [n x k] and alpha is [k x 1]; x is stored columnwise
      void mass_axpy(resolveInt size, resolveReal* alpha, resolveReal k, resolveReal* x,resolveReal* y);

      //mass dot: V^T x, where V is [n x k] and x is [k x 2], everything is stored and returned columnwise
      resolveReal* mass_dot(resolveReal size, resolveReal* V, resolveReal k, resolveReal* x);

    private:
      resolveLinAlgWorkspace* workspace_;
  };

}
