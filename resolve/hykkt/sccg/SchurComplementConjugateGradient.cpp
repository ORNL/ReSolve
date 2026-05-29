#include "SchurComplementConjugateGradient.hpp"

#include <cmath>

#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    /** Constructor for SchurComplementConjugateGradient.
     *  @param n[in] - Dimension of outer system.
     *  @param m[in] - Dimension of inner system.
     *  @param choleskySolver[in] - Factorization of Hgamma to use for direct solve.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     */
    SchurComplementConjugateGradient::SchurComplementConjugateGradient(
        index_type          n,
        index_type          m,
        CholeskySolver*     choleskySolver,
        memory::MemorySpace memspace,
        MatrixHandler&      matrix_handler,
        VectorHandler&      vector_handler)
      : n_(n),
        m_(m),
        matrix_handler_(matrix_handler),
        vector_handler_(vector_handler),
        choleskySolver_(choleskySolver),
        y_(m_),
        z_(m_),
        r_(n_),
        p_(n_),
        s_(n_),
        w_(n_),
        memspace_(memspace)
    {
      ;
    }

    /**
     * @brief Loads or reloads matrix pointers to the solver
     * @param[in] jc - Pointer to the JC matrix in CSR format.
     * @param[in] jc_tr - Pointer to the transposed JC matrix in CSR format.
     */
    void SchurComplementConjugateGradient::addMatrixInfo(matrix::Csr* jc, matrix::Csr* jc_tr)
    {
      jc_    = jc;
      jc_tr_ = jc_tr;
    }

    /**
     * @brief Loads or reloads vector pointers to the solver
     * @param[in] x0 - Pointer to the left-hand side vector.
     * @param[in] b - Pointer to the right-hand side vector.
     */
    void SchurComplementConjugateGradient::addVectorInfo(vector::Vector* x0, vector::Vector* b)
    {
      x0_ = x0;
      b_  = b;
    }

    /**
     * @brief Reloads pointer to the Cholesky solver
     * @param[in] choleskySolver - Factorization of Hgamma to use for direct solve.
     */
    void SchurComplementConjugateGradient::updateCholeskySolver(CholeskySolver* choleskySolver)
    {
      choleskySolver_ = choleskySolver;
    }

    void SchurComplementConjugateGradient::setSolverTolerance(double tol)
    {
      tol_ = tol;
    }

    void SchurComplementConjugateGradient::setSolverItmax(int itmax)
    {
      itmax_ = itmax;
    }

    void SchurComplementConjugateGradient::setup()
    {
      y_.allocate(memspace_);
      z_.allocate(memspace_);
      r_.allocate(memspace_);
      p_.allocate(memspace_);
      s_.allocate(memspace_);
      w_.allocate(memspace_);

      y_.setToZero(memspace_);
      z_.setToZero(memspace_);
      r_.setToZero(memspace_);
      p_.setToZero(memspace_);
      s_.setToZero(memspace_);
      w_.setToZero(memspace_);

      beta_ = 0;
    }

    int SchurComplementConjugateGradient::solve()
    {
      using namespace constants;

      matrix_handler_.matvec(jc_tr_, x0_, &y_, &ONE, &ZERO, memspace_);
      choleskySolver_->solve(&z_, &y_);
      matrix_handler_.matvec(jc_, &z_, &r_, &MINUS_ONE, &ONE, memspace_);
      gam_i_ = vector_handler_.dot(&r_, &r_, memspace_);

      matrix_handler_.matvec(jc_tr_, &r_, &y_, &ONE, &ZERO, memspace_);
      choleskySolver_->solve(&z_, &y_);
      matrix_handler_.matvec(jc_, &z_, &w_, &ONE, &ZERO, memspace_);
      delta_ = vector_handler_.dot(&w_, &r_, memspace_);
      alpha_ = gam_i_ / delta_;

      int i;
      for (i = 0; i < itmax_; i++)
      {
        vector_handler_.scal(beta_, &p_, memspace_);
        vector_handler_.axpy(ONE, &r_, &p_, memspace_);
        vector_handler_.scal(beta_, &s_, memspace_);
        vector_handler_.axpy(ONE, &w_, &s_, memspace_);
        vector_handler_.axpy(alpha_, &p_, x0_, memspace_);
        minalpha_ = -alpha_;
        vector_handler_.axpy(minalpha_, &s_, &r_, memspace_);
        gam_i1_ = vector_handler_.dot(&r_, &r_, memspace_);
        if (sqrt(gam_i1_) < tol_)
        {
          printf("Convergence occured at iteration %d\n", i);
          break;
        }
        matrix_handler_.matvec(jc_tr_, &r_, &y_, &ONE, &ZERO, memspace_);
        choleskySolver_->solve(&z_, &y_);
        matrix_handler_.matvec(jc_, &z_, &w_, &ONE, &ZERO, memspace_);
        delta_ = vector_handler_.dot(&w_, &r_, memspace_);
        beta_  = gam_i1_ / gam_i_;
        gam_i_ = gam_i1_;
        alpha_ = gam_i_ / (delta_ - beta_ * gam_i_ / alpha_);
      }

      printf("Error is %32.32g \n", sqrt(gam_i1_));
      if (i == itmax_)
      {
        printf("No CG convergence in %d iterations\n", itmax_);
        return 1;
      }

      return 0;
    }

  } // namespace hykkt
} // namespace ReSolve
