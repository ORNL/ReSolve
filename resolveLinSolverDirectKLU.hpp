
#include "klu.h"
#include "resolveCommon.hpp"
#include "resolveMatrix.hpp"

namespace ReSolve {
  class resolveLinSolverDirectKLU : public resolveLinSolverDirect {
    public:
      resolveLinSolverDirectKLU();
      ~resolveLinSolverDirectKLU();
      void setup(resolveMatrix* A);
     
      void setupParameters(int ordering, double KLU_threshold, bool halt_if_singular);


      void analyze(); //the same as symbolic factorization
      void factorize();
      void refactorize();
      resolveReal* solve(resolveReal* rhs); 
    
    private:

      klu_common* common; //settings
      klu_symbolic* Symbolic;
      klu_numeric* Numeric, 
  };
}
