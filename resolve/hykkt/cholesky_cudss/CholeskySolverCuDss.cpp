/**
 * @file CholeskySolverCuDss.cpp
 * @brief Cholesky decomposition solver CuDSS implementation. This is a CUDA-only variant of CholeskySolver.
 */

#include "CholeskySolverCuDss.hpp"

#include "CholeskySolverCuDssCuda.hpp"

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
    CholeskySolverCuDss::CholeskySolverCuDss(memory::MemorySpace memspace)
      : memspace_(memspace),
        impl_(new CholeskySolverCuDssCuda())
    {
    }

    /**
     * @brief Cholesky Solver destructor
     */
    CholeskySolverCuDss::~CholeskySolverCuDss()
    {
      delete impl_;
    }

    /**
     * @brief Loads or reloads matrix pointer to the solver
     * @param[in] A - pointer to the matrix in CSR format
     */
    void CholeskySolverCuDss::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
      impl_->addMatrixInfo(A);
    }

    /**
     * @brief Performs symbolic analysis. Determines the sparsity pattern of
     *        the factor L. Values will be computed by numerical analysis.
     *        This need only be called once as long as the sparsity pattern does not change.
     */
    void CholeskySolverCuDss::symbolicAnalysis()
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
    void CholeskySolverCuDss::setPivotTolerance(real_type tol)
    {
      tol_ = tol;
    }

    /**
     * @brief Performs numerical factorization. Fills in the values of the factor L such
     *        that LL^T = A.
     */
    void CholeskySolverCuDss::numericalFactorization()
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
    void CholeskySolverCuDss::solve(vector::Vector* x, vector::Vector* b)
    {
      impl_->solve(x, b);
      x->setDataUpdated(memspace_);
    }
  } // namespace hykkt
} // namespace ReSolve
