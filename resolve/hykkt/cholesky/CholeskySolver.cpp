#include "CholeskySolver.hpp"

namespace ReSolve {
  namespace hykkt {
    CholeskySolver::CholeskySolver(memory::MemorySpace memspace):
      memspace_(memspace),
    {
      if (memspace_ == memory::HOST)
      {
        impl_ = CholeskySolverCPU();
      }
      else
      {
#ifdef RESOLVE_USE_CUDA
        impl_ = CholeskySolverCuda();
#elif defined(RESOLVE_USE_HIP)
        impl_ = CholeskySolverHip();
#else
        out::error() << "No GPU support enabled, and memory space set to DEVICE.\n";
        exit(1);
#endif
      }
    }

    CholeskySolver::~CholeskySolver()
    {}

    void CholeskySolver::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
    }

    void CholeskySolver::symbolicAnalysis()
    {
      impl_.symbolicAnalysis(A_);
    }

    void CholeskySolver::setPivotTolerance(real_type tol)
    {
      tol_ = tol;
    }

    void CholeskySolver::numericalFactorization()
    {
      impl_.numericalFactorization(A_, tol_);
    }
    
    void CholeskySolver::solve(vector::Vector* x, vector::Vector* b)
    {
      impl_.solve(x, b);
    }
  } // namespace hykkt
} // namespace ReSolve