#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve {
  namespace hykkt {
    class CholeskySolver {
    public:
      CholeskySolver();
      ~CholeskySolver();

      void addMatrixInfo(matrix::Csr* A);
      void symbolicAnalysis();
      void setPivotTolerance(real_type tol);
      void numericalFactorization();
      void solve(vector::Vector* x, vector::Vector* b);

    private:
      memory::MemorySpace memspace_;

      matrix::Csr* A_;
      real_type tol_ = 1e-12;
      CholeskySolverImpl impl_;
    };
  }
}