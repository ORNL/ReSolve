#include <cholmod.h>

#include "CholeskySolverImpl.hpp"

namespace ReSolve
{
  namespace hykkt
  {
    class CholeskySolverCpu : public CholeskySolverImpl
    {
    public:
      CholeskySolverCpu();
      ~CholeskySolverCpu();

      void addMatrixInfo(matrix::Csr* A);
      void symbolicAnalysis();
      void numericalFactorization(real_type tol);
      void solve(vector::Vector* x, vector::Vector* b);

    private:
      MemoryHandler mem_;

      cholmod_common  Common_;
      cholmod_sparse* A_chol_; // cholmod sparse matrix representation
      cholmod_factor* factorization_;

      // helper methods to convert between ReSolve and cholmod types
      cholmod_sparse* convertToCholmod(matrix::Csr* A);
      cholmod_dense*  convertToCholmod(vector::Vector* v);
    };
  } // namespace hykkt
} // namespace ReSolve
