#pragma once

#include "klu.h"

#include "Common.hpp"
#include <resolve/LinSolverDirect.hpp>

namespace ReSolve 
{
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  // Forward declaration of matrix::Sparse class
  namespace matrix
  {
    class Sparse;
  }

  class LinSolverDirectKLU : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public:
      LinSolverDirectKLU();
      ~LinSolverDirectKLU();

      int setup(matrix::Sparse* A,
                matrix::Sparse* L = nullptr,
                matrix::Sparse* U = nullptr,
                index_type*     P = nullptr,
                index_type*     Q = nullptr,
                vector_type*  rhs = nullptr) override;
     
      int analyze() override; //the same as symbolic factorization
      int factorize() override;
      int refactorize() override;
      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* x) override;
    
      matrix::Sparse* getLFactor() override; 
      matrix::Sparse* getUFactor() override; 
      index_type*  getPOrdering() override;
      index_type*  getQOrdering() override;

      virtual void setPivotThreshold(real_type tol) override;
      virtual void setOrdering(int ordering) override;
      virtual void setHaltIfSingular(bool isHalt) override;

      virtual real_type getMatrixConditionNumber() override;

    private:
      bool factors_extracted_{false};
      klu_common Common_; //settings
      klu_symbolic* Symbolic_{nullptr};
      klu_numeric* Numeric_{nullptr}; 
  };
}
