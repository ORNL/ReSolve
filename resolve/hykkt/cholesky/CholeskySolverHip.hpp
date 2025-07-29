#include "CholeskySolverImpl.hpp"

namespace ReSolve
{
  namespace hykkt
  {
    class CholeskySolverHip : public CholeskySolverImpl
    {
    public:
      CholeskySolverHip();
      ~CholeskySolverHip();

      void addMatrixInfo(matrix::Csr* A);
      void symbolicAnalysis();
      void numericalFactorization(real_type tol);
      void solve(vector::Vector* x, vector::Vector* b);

    private:
      MemoryHandler mem_;

      matrix::Csr* A_; // pointer to the input matrix
    };
  } // namespace hykkt
} // namespace ReSolve
