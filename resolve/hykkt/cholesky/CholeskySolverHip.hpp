/**
 * @file CholeskySolverHip.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Header for HIP implementation of Cholesky Solver
 */

#include <cholmod.h>

#include <rocsolver/rocsolver.h>

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

      cholmod_common  Common_;
      cholmod_sparse* A_chol_; // cholmod sparse matrix representation
      cholmod_factor* factorization_;

      rocblas_handle   handle_;
      rocsolver_rfinfo rfinfo_;

      matrix::Csr* A_;
      matrix::Csr* L_;
      index_type*  Q_;

      real_type*   rhs_tmp_;// used during analysis
 
      cholmod_sparse* convertToCholmod(matrix::Csr* A);
    };
  } // namespace hykkt
} // namespace ReSolve
