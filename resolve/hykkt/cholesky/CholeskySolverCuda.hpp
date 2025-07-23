#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include "CholeskySolverImpl.hpp"

namespace ReSolve {
  namespace hykkt {
    class CholeskySolverCuda : public CholeskySolverImpl {
    public:
      CholeskySolverCuda();
      ~CholeskySolverCuda();

      void symbolicAnalysis(matrix::Csr* A);
      void numericalFactorization(matrix::Csr* A, real_type tol);
      void solve(matrix::Csr* A, vector::Vector* x, vector::Vector* b);
    private:
      MemoryHandler mem_;
        //handle to the cuSPARSE library context
      cusolverSpHandle_t cusolverHandle_;
      cusparseMatDescr_t descrA_; //descriptor for matrix A 
      csrcholInfo_t factorizationInfo_; // stores Cholesky factorization
      void* buffer_; // buffer for Cholesky factorization
    };
  }
}