#include "ConjugateGradient.hpp"

#include <cmath>
#include <chrono>

#include <resolve/Common.hpp>

#include <cuda_runtime.h>

namespace ReSolve
{
  namespace hykkt
  {
    /** Constructor for ConjugateGradient.
     *  @param n[in] - Dimension of outer system.
     *  @param m[in] - Dimension of inner system.
     *  @param choleskySolver[in] - Factorization of H_gamma to use for direct solve.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     */
    ConjugateGradient::ConjugateGradient(
        index_type          n,
        MatrixHandler*      matrix_handler,
        VectorHandler*      vector_handler,
        memory::MemorySpace memspace)
      : n_(n),
        matrix_handler_(matrix_handler),
        vector_handler_(vector_handler),
        memspace_(memspace)
    {
      ;
    }

    ConjugateGradient::~ConjugateGradient()
    {
      delete A_prec_;
      delete r_;
      delete r_prec_;
      delete b_prec_;
      delete p_;
      delete s_;
      delete w_;
      delete impl_;
    }

    /**
     * @brief Loads or reloads matrix pointers to the solver
     * @param[in] J - Pointer to the JC matrix in CSR format.
     * @param[in] J_tr - Pointer to the transposed JC matrix in CSR format.
     */
    void ConjugateGradient::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
    }

    /**
     * @brief Loads or reloads vector pointers to the solver
     * @param[in] x_0 - Pointer to the left-hand side vector.
     * @param[in] b - Pointer to the right-hand side vector.
     */
    void ConjugateGradient::addVectorInfo(vector::Vector* x_0, vector::Vector* b)
    {
      x_0_ = x_0;
      b_   = b;
    }

    /**
     * @brief Loads or reloads preconditioner matrix pointers to the solver. This transforms
     * the system into L^-1 * A * L^-T * y = L * b.
     * @param[in] L - Pointer to the lower triangular preconditioner matrix (L) in CSR format.
     * @param[in] L_tr_ - Pointer to the transpose preconditioner matrix (L^T) in CSR format.
     */
    void ConjugateGradient::addPreconditionerInfo(vector::Vector* d_inv)
    {
      d_inv_ = d_inv;
    }

    void ConjugateGradient::setSolverTolerance(double tol)
    {
      tol_ = tol;
    }

    void ConjugateGradient::setSolverItmax(int itmax)
    {
      itmax_ = itmax;
    }

    void ConjugateGradient::setup()
    {
      impl_ = new RandomizedConjugateGradientCuda(vector_handler_);

      A_prec_ = new matrix::Csr(n_, n_, A_->getNnz());
      r_ = new vector::Vector(n_);
      r_prec_ = new vector::Vector(n_);
      b_prec_ = new vector::Vector(n_);
      p_ = new vector::Vector(n_);
      s_ = new vector::Vector(n_);
      w_ = new vector::Vector(n_);

      A_prec_->allocateMatrixData(memspace_);
      r_->allocate(memspace_);
      r_prec_->allocate(memspace_);
      b_prec_->allocate(memspace_);
      p_->allocate(memspace_);
      s_->allocate(memspace_);
      w_->allocate(memspace_);

      A_prec_->copyFromExternal(A_->getRowData(memspace_),
                                A_->getColData(memspace_),
                                A_->getValues(memspace_),
                                memspace_,
                                memspace_);
      r_->copyFromExternal(b_, memspace_, memspace_);
      b_prec_->copyFromExternal(b_, memspace_, memspace_);
      p_->copyFromExternal(b_, memspace_, memspace_);
      s_->copyFromExternal(b_, memspace_, memspace_);
      w_->copyFromExternal(b_, memspace_, memspace_);

      x_0_->setToZero(memspace_);

      beta_ = 0;

      b_norm_ = std::sqrt(vector_handler_->dot(b_, b_, memspace_));
      
      impl_->setup(1);
    }

    void ConjugateGradient::precondition()
    {
      using namespace constants;

      // A_prec = L^-1 * A * L^-T
      // variable names are a bit messed up
      matrix_handler_->leftScale(d_inv_, A_prec_, memspace_);
      matrix_handler_->rightScale(A_prec_, d_inv_, memspace_);

      // b_prec = 1 / b_norm * L^-1 * b
      b_prec_->copyFromExternal(b_, memspace_, memspace_);
      vector_handler_->scal(d_inv_, b_prec_, memspace_);
      vector_handler_->scal(1.0 / b_norm_, b_prec_, memspace_);
    }

    int ConjugateGradient::solve()
    {
      using namespace constants;

      precondition();
      r_prec_->copyFromExternal(b_prec_, memspace_, memspace_);

      matrix_handler_->matvec(A_prec_, x_0_, r_prec_, &MINUS_ONE, &ONE, memspace_);
      gamma_i_ = vector_handler_->dot(r_prec_, r_prec_, memspace_);

      matrix_handler_->matvec(A_prec_, r_prec_, w_, &ONE, &ZERO, memspace_);
      // impl_->SpMMTallSkinny(A_prec_, r_prec_, w_); // IS THIS FASTER???
      delta_ = vector_handler_->dot(w_, r_prec_, memspace_);
      alpha_ = gamma_i_ / delta_;

      int i;
      for (i = 0; i < itmax_; i++)
      {
      auto start = std::chrono::steady_clock::now();
        vector_handler_->scal(beta_, p_, memspace_);
        vector_handler_->axpy(ONE, r_prec_, p_, memspace_);
        vector_handler_->scal(beta_, s_, memspace_);
        vector_handler_->axpy(ONE, w_, s_, memspace_);
        vector_handler_->axpy(alpha_, p_, x_0_, memspace_);
        vector_handler_->axpy(-alpha_, s_, r_prec_, memspace_);
        gamma_i1_ = vector_handler_->dot(r_prec_, r_prec_, memspace_);

        r_->copyFromExternal(r_prec_, memspace_, memspace_);
        vector_handler_->diagSolve(d_inv_, r_, memspace_);
        vector_handler_->scal(b_norm_, r_, memspace_); // can maybe save one operation in computing error
        r_norm_ = std::sqrt(vector_handler_->dot(r_, r_, memspace_));
        error_ = r_norm_ / b_norm_;
        if (error_ < tol_)
        {
          // auto end = std::chrono::steady_clock::now();
          // std::chrono::duration<double, std::milli> elapsed = (end - start);
          // printf("Convergence occured at iteration %d. Took %f ms.\n", i, elapsed.count());
          break;
        }
        matrix_handler_->matvec(A_prec_, r_prec_, w_, &ONE, &ZERO, memspace_);
        // impl_->SpMMTallSkinny(A_prec_, r_prec_, w_);
        delta_   = vector_handler_->dot(w_, r_prec_, memspace_);
        beta_    = gamma_i1_ / gamma_i_;
        gamma_i_ = gamma_i1_;
        alpha_   = gamma_i_ / (delta_ - beta_ * gamma_i_ / alpha_);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("time = %f, error = %f\n", elapsed.count(), error_);
      }

      printf("Conjugate gradient error is %32.32g \n", error_);
      if (i == itmax_)
      {
        printf("No CG convergence in %d iterations\n", itmax_);
        return 1;
      }
      return 0;
    }

  } // namespace hykkt
} // namespace ReSolve
