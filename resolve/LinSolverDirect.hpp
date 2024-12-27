/**
 * @file LinSolverIterative.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of direct solver base class.
 * 
 */
#pragma once

#include <string>
#include <resolve/LinSolver.hpp>

namespace ReSolve 
{
  class LinSolverDirect : public LinSolver 
  {
    public:
      LinSolverDirect();
      virtual ~LinSolverDirect();
      virtual int setup(matrix::Sparse* A = nullptr,
                        matrix::Sparse* L = nullptr,
                        matrix::Sparse* U = nullptr,
                        index_type*     P = nullptr,
                        index_type*     Q = nullptr,
                        vector_type*  rhs = nullptr);

      virtual int analyze(); //the same as symbolic factorization
      virtual int factorize();
      virtual int refactorize();
      virtual int solve(vector_type* rhs, vector_type* x) = 0;
      virtual int solve(vector_type* x) = 0;
     
      virtual matrix::Sparse* getLFactor(); 
      virtual matrix::Sparse* getUFactor(); 
      virtual index_type*  getPOrdering();
      virtual index_type*  getQOrdering();

      virtual void setPivotThreshold(real_type tol);
      virtual void setOrdering(int ordering);
      virtual void setHaltIfSingular(bool is_halt);

      virtual real_type getMatrixConditionNumber();
    
    protected:
      matrix::Sparse* L_{nullptr};
      matrix::Sparse* U_{nullptr};
      index_type* P_{nullptr};
      index_type* Q_{nullptr};

      int ordering_{1}; // 0 = AMD, 1 = COLAMD, 2 = user provided P, Q
      real_type pivot_threshold_tol_{0.1};
      bool halt_if_singular_{false};
  };

}
