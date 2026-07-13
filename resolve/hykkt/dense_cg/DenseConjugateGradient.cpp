#include "DenseConjugateGradient.hpp"

#include <cmath>
#include <chrono>

#include <resolve/Common.hpp>

// #include <cuda_runtime.h>

namespace ReSolve
{
  namespace hykkt
  {
    /** Constructor for DenseConjugateGradient.
     *  @param n[in] - Dimension of outer system.
     *  @param m[in] - Dimension of inner system.
     *  @param choleskySolver[in] - Factorization of H_gamma to use for direct solve.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     */
    DenseConjugateGradient::DenseConjugateGradient(
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

    DenseConjugateGradient::~DenseConjugateGradient()
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
    void DenseConjugateGradient::addMatrixInfo(vector::Vector* A)
    {
      A_ = A;
    }

    /**
     * @brief Loads or reloads vector pointers to the solver
     * @param[in] x_0 - Pointer to the left-hand side vector.
     * @param[in] b - Pointer to the right-hand side vector.
     */
    void DenseConjugateGradient::addVectorInfo(vector::Vector* x_0, vector::Vector* b)
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
    void DenseConjugateGradient::addPreconditionerInfo(vector::Vector* d, vector::Vector* d_inv)
    {
      d_ = d;
      d_inv_ = d_inv;
    }

    void DenseConjugateGradient::setSolverTolerance(double tol)
    {
      tol_ = tol;
    }

    void DenseConjugateGradient::setSolverItmax(int itmax)
    {
      itmax_ = itmax;
    }

    void DenseConjugateGradient::setup()
    {
      impl_ = new RandomizedConjugateGradientCuda(vector_handler_);

      A_prec_ = new vector::Vector(n_, n_);
      r_ = new vector::Vector(n_);
      r_prec_ = new vector::Vector(n_);
      b_prec_ = new vector::Vector(n_);
      p_ = new vector::Vector(n_);
      s_ = new vector::Vector(n_);
      w_ = new vector::Vector(n_);

      A_prec_->allocate(memspace_);
      r_->allocate(memspace_);
      r_prec_->allocate(memspace_);
      b_prec_->allocate(memspace_);
      p_->allocate(memspace_);
      s_->allocate(memspace_);
      w_->allocate(memspace_);

      beta_ = 0;
      
      impl_->setup(1);
    }

    void DenseConjugateGradient::precondition()
    {
      using namespace constants;

      // A_prec = L^-1 * A * L^-T
      // variable names are a bit messed up
      A_prec_->copyFromExternal(A_, memspace_, memspace_);
      impl_->preconditionDense(A_prec_, d_inv_);

      // b_prec = 1 / b_norm * L^-1 * b
      b_prec_->copyFromExternal(b_, memspace_, memspace_);
      vector_handler_->scal(d_inv_, b_prec_, memspace_);
      vector_handler_->scal(1.0 / b_norm_, b_prec_, memspace_);
    }

    int DenseConjugateGradient::solve()
    {
      using namespace constants;
      auto start = std::chrono::steady_clock::now();

      b_norm_ = vector_handler_->norm(b_, memspace_);

      precondition();

      x_0_->setToZero(memspace_);
      // vector_handler_->randomVector(x_0_, -1.0, 1.0, memspace_);
      // vector_handler_->gemm('N', 'N', ONE, ZERO, A_prec_, x_0_, r_prec_, memspace_);
      // real_type AX_prec_0_norm = vector_handler_->norm(r_prec_, memspace_);
      // real_type B_prec_norm = vector_handler_->norm(b_prec_, memspace_);
      // real_type normalization_factor = B_prec_norm / AX_prec_0_norm;
      // vector_handler_->scal(normalization_factor, x_0_, memspace_);
      r_prec_->copyFromExternal(b_prec_, memspace_, memspace_);

      vector_handler_->gemm('N', 'N', MINUS_ONE, ONE, A_prec_, x_0_, r_prec_, memspace_);
      gamma_i_ = vector_handler_->dot(r_prec_, r_prec_, memspace_);

      vector_handler_->gemm('N', 'N', ONE, ZERO, A_prec_, r_prec_, w_, memspace_);
      delta_ = vector_handler_->dot(w_, r_prec_, memspace_);
      alpha_ = gamma_i_ / delta_;

      int i;
      for (i = 0; i < itmax_; i++)
      {
      // auto start = std::chrono::steady_clock::now();
        vector_handler_->scal(beta_, p_, memspace_);
        vector_handler_->axpy(ONE, r_prec_, p_, memspace_);
        vector_handler_->scal(beta_, s_, memspace_);
        vector_handler_->axpy(ONE, w_, s_, memspace_);
        vector_handler_->axpy(alpha_, p_, x_0_, memspace_);
        vector_handler_->axpy(-alpha_, s_, r_prec_, memspace_);
        gamma_i1_ = vector_handler_->dot(r_prec_, r_prec_, memspace_);

        r_->copyFromExternal(r_prec_, memspace_, memspace_);
        vector_handler_->scal(d_, r_, memspace_);
        vector_handler_->scal(b_norm_, r_, memspace_); // can maybe save one operation in computing error
        r_norm_ = std::sqrt(vector_handler_->dot(r_, r_, memspace_));
        error_ = r_norm_ / b_norm_;
        // printf("%.7e\n", error_);
        if (error_ < tol_)
        {
          auto end = std::chrono::steady_clock::now();
          std::chrono::duration<double, std::milli> elapsed = (end - start);
          printf("Convergence occured at iteration %d. Took %f ms.\n", i, elapsed.count());
          break;
        }
        vector_handler_->gemm('N', 'N', ONE, ZERO, A_prec_, r_prec_, w_, memspace_);
        delta_   = vector_handler_->dot(w_, r_prec_, memspace_);
        beta_    = gamma_i1_ / gamma_i_;
        gamma_i_ = gamma_i1_;
        alpha_   = gamma_i_ / (delta_ - beta_ * gamma_i_ / alpha_);
        // auto end = std::chrono::steady_clock::now();
        // std::chrono::duration<double, std::milli> elapsed = (end - start);
        // printf("time = %f, error = %f\n", elapsed.count(), error_);
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
