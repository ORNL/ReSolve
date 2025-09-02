/**
 * @file CholeskySolverCpu.cpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief CPU implementation of Cholesky Solver
 */

#include "CholeskySolverCpu.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    CholeskySolverCpu::CholeskySolverCpu()
    {
      Common_.nmethods = 1;
      // Use natural ordering
      Common_.method[0].ordering = CHOLMOD_NATURAL;

      A_chol_        = nullptr;
      factorization_ = nullptr;
      cholmod_start(&Common_);
    }

    CholeskySolverCpu::~CholeskySolverCpu()
    {
      if (A_chol_)
      {
        cholmod_free_sparse(&A_chol_, &Common_);
      }
      if (factorization_)
      {
        cholmod_free_factor(&factorization_, &Common_);
      }
      cholmod_finish(&Common_);
    }

    void CholeskySolverCpu::addMatrixInfo(matrix::Csr* A)
    {
      if (A_chol_)
      {
        cholmod_free_sparse(&A_chol_, &Common_);
      }
      A_chol_ = convertToCholmod(A);
    }

    /**
     * @brief Performs symbolic analysis
     *
     * Uses the `cholmod_analyze` routine.
     */
    void CholeskySolverCpu::symbolicAnalysis()
    {
      factorization_ = cholmod_analyze(A_chol_, &Common_);
      if (Common_.status < 0)
      {
        out::error() << "Cholesky symbolic analysis failed with status: " << Common_.status << "\n";
      }
    }

    /**
     * @brief Performs symbolic analysis
     * @brief Performs numerical factorization
     *
     * Uses the `cholmod_factorize` routine.
     *
     * @param[in] tol - This is ignored in the CPU implementation
     */
    void CholeskySolverCpu::numericalFactorization(real_type tol)
    {
      (void) tol; // Mark tol as unused

      cholmod_factorize(A_chol_, factorization_, &Common_);
      if (Common_.status < 0)
      {
        out::error() << "Cholesky factorization failed with status: " << Common_.status << "\n";
      }
    }

    /**
     * @brief Solves the system Ax = b and stores the result in x.
     *
     * Uses the `cholmod_solve` routine.
     *
     * @param[out] x - The solution vector.
     * @param[in]  b - The right-hand side vector.
     */
    void CholeskySolverCpu::solve(vector::Vector* x, vector::Vector* b)
    {
      cholmod_dense* b_chol = convertToCholmod(b);
      cholmod_dense* x_chol = cholmod_solve(CHOLMOD_A, factorization_, b_chol, &Common_);
      if (Common_.status < 0)
      {
        out::error() << "Cholesky solve failed with status: " << Common_.status << "\n";
      }
      x->copyDataFrom(static_cast<real_type*>(x_chol->x), memory::HOST, memory::HOST);
    }

    /**
     * @brief Converts a ReSolve type CSR matrix to CHOLMOD format.
     *
     * @param[in] A - pointer to the CSR matrix
     * @return Pointer to the equivalent CHOLMOD sparse matrix
     */
    cholmod_sparse* CholeskySolverCpu::convertToCholmod(matrix::Csr* A)
    {
      A_chol_ = cholmod_allocate_sparse((size_t) A->getNumRows(),
                                        (size_t) A->getNumColumns(),
                                        (size_t) A->getNnz(),
                                        1,
                                        1,
                                        1,
                                        CHOLMOD_REAL,
                                        &Common_);
      mem_.copyArrayHostToHost(
          static_cast<int*>(A_chol_->p), A->getRowData(memory::HOST), A->getNumRows() + 1);
      mem_.copyArrayHostToHost(
          static_cast<int*>(A_chol_->i), A->getColData(memory::HOST), A->getNnz());
      mem_.copyArrayHostToHost(
          static_cast<double*>(A_chol_->x), A->getValues(memory::HOST), A->getNnz());

      return A_chol_;
    }

    /**
     * @brief Converts a ReSolve type vector to CHOLMOD format.
     *
     * @param[in] v - pointer to the vector
     * @return Pointer to the equivalent CHOLMOD dense matrix
     */
    cholmod_dense* CholeskySolverCpu::convertToCholmod(vector::Vector* v)
    {
      cholmod_dense* v_chol = cholmod_allocate_dense((size_t) v->getSize(),
                                                     1,
                                                     (size_t) v->getSize(),
                                                     CHOLMOD_REAL,
                                                     &Common_);
      mem_.copyArrayHostToHost(
          static_cast<double*>(v_chol->x), v->getData(memory::HOST), v->getSize());
      return v_chol;
    }
  } // namespace hykkt
} // namespace ReSolve
