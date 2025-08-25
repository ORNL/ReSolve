/** 
 * @file CholeskySolver.cpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Cholesky decomposition solver implementation
 */

#include "CholeskySolver.hpp"

#include "CholeskySolverCpu.hpp"
#ifdef RESOLVE_USE_CUDA
#include "CholeskySolverCuda.hpp"
#elif defined(RESOLVE_USE_HIP)
#include "CholeskySolverHip.hpp"
#endif

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    /**
     * @brief Cholesky Solver constructor
     * @param[in] memspace - memory space to use for computations
     */
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

    /** 
     * @brief Cholesky Solver destructor
     */
    CholeskySolver::~CholeskySolver()
    {
      delete impl_;
    }

    /**
     * @brief Loads or reloads matrix pointer to the solver
     * @param[in] A - pointer to the matrix in CSR format
     */
    void CholeskySolver::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
      impl_->addMatrixInfo(A);
    }

    /**
     * @brief Performs symbolic analysis. This need only be called once
     *        as long as the sparsity pattern does not change.
     */
    void CholeskySolver::symbolicAnalysis()
    {
      impl_->symbolicAnalysis();
    }

    /**
     * @brief Sets the pivot tolerance for the solver.
     *
     * This is only used in the CUDA implementation. For other backends,
     * it is ignored.
     * 
     * @param[in] tol - pivot tolerance value
     */
    void CholeskySolver::setPivotTolerance(real_type tol)
    {
      tol_ = tol;
    }

    /**
     * @brief Performs numerical factorization.
     */
    void CholeskySolver::numericalFactorization()
    {
      impl_->numericalFactorization(tol_);
    }

    /**
     * @brief Solves the linear system Ax = b and stores the result in x.
     *
     * @pre The vector x is allocated in the given memspace.
     *
     * @param[out] x - pointer to the solution vector
     * @param[in] b - pointer to the right-hand side vector
     */
    void CholeskySolver::solve(vector::Vector* x, vector::Vector* b)
    {
      impl_->solve(x, b);
      x->setDataUpdated(memspace_);
    }
  } // namespace hykkt
} // namespace ReSolve
