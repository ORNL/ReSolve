#include "CholeskySolver.hpp"

#include "CholeskySolverCpu.hpp"
#ifdef RESOLVE_USE_CUDA
#include "CholeskySolverCuda.hpp"
#elif defined(RESOLVE_USE_HIP)
#include "CholeskySolverHip.hpp"
#endif

namespace ReSolve
{
  namespace hykkt
  {
    CholeskySolver::CholeskySolver(memory::MemorySpace memspace)
      : memspace_(memspace)
    {
      if (memspace_ == memory::HOST)
      {
        impl_ = new CholeskySolverCpu();
      }
      else
      {
#ifdef RESOLVE_USE_CUDA
        impl_ = new CholeskySolverCuda();
#elif defined(RESOLVE_USE_HIP)
        impl_ = new CholeskySolverHip();
#else
        out::error() << "No GPU support enabled, and memory space set to DEVICE.\n";
        exit(1);
#endif
      }
    }

    CholeskySolver::~CholeskySolver()
    {
    }

    void CholeskySolver::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
      impl_->addMatrixInfo(A);
    }

    void CholeskySolver::symbolicAnalysis()
    {
      impl_->symbolicAnalysis();
    }

    void CholeskySolver::setPivotTolerance(real_type tol)
    {
      tol_ = tol;
    }

    void CholeskySolver::numericalFactorization()
    {
      impl_->numericalFactorization(tol_);
    }

    void CholeskySolver::solve(vector::Vector* x, vector::Vector* b)
    {
      impl_->solve(x, b);
      x->setDataUpdated(memspace_);
    }
  } // namespace hykkt
} // namespace ReSolve
