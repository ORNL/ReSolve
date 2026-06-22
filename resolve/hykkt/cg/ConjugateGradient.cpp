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
      delete r_;
      delete p_;
      delete s_;
      delete w_;
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
      r_ = new vector::Vector(n_);
      p_ = new vector::Vector(n_);
      s_ = new vector::Vector(n_);
      w_ = new vector::Vector(n_);

      r_->allocate(memspace_);
      p_->allocate(memspace_);
      s_->allocate(memspace_);
      w_->allocate(memspace_);

      r_->copyFromExternal(b_, memspace_, memspace_);
      p_->copyFromExternal(b_, memspace_, memspace_);
      s_->copyFromExternal(b_, memspace_, memspace_);
      w_->copyFromExternal(b_, memspace_, memspace_);

      x_0_->setToZero(memspace_);

      beta_ = 0;
    }

    int ConjugateGradient::solve()
    {
      using namespace constants;
      matrix_handler_->matvec(A_, x_0_, r_, &ONE, &ZERO, memspace_);
      gamma_i_ = vector_handler_->dot(r_, r_, memspace_);

      matrix_handler_->matvec(A_, r_, w_, &ONE, &ZERO, memspace_);
      delta_ = vector_handler_->dot(w_, r_, memspace_);
      alpha_ = gamma_i_ / delta_;

      int i;
      for (i = 0; i < itmax_; i++)
      {
// cudaEvent_t start, stop;
// cudaEventCreate(&start);
// cudaEventCreate(&stop);
// cudaEventRecord(start, 0);
        auto start = std::chrono::steady_clock::now();
        vector_handler_->scal(beta_, p_, memspace_);
        vector_handler_->axpy(ONE, r_, p_, memspace_);
        vector_handler_->scal(beta_, s_, memspace_);
        vector_handler_->axpy(ONE, w_, s_, memspace_);
        vector_handler_->axpy(alpha_, p_, x_0_, memspace_);
        vector_handler_->axpy(-alpha_, s_, r_, memspace_);
        gamma_i1_ = vector_handler_->dot(r_, r_, memspace_);
// cudaEventRecord(stop, 0);
// cudaEventSynchronize(stop);
// float ms = 0;
// cudaEventElapsedTime(&ms, start, stop);
// std::cout << ms << '\n';
        if (sqrt(gamma_i1_) < tol_)
        {
          printf("Convergence occured at iteration %d\n", i);
          break;
        }
        matrix_handler_->matvec(A_, r_, w_, &ONE, &ZERO, memspace_);
        delta_   = vector_handler_->dot(w_, r_, memspace_);
        beta_    = gamma_i1_ / gamma_i_;
        gamma_i_ = gamma_i1_;
        alpha_   = gamma_i_ / (delta_ - beta_ * gamma_i_ / alpha_);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("time = %f\n", elapsed.count());
      }

      printf("Conjugate gradient error is %32.32g \n", sqrt(gamma_i1_));
      if (i == itmax_)
      {
        printf("No CG convergence in %d iterations\n", itmax_);
        return 1;
      }
      return 0;
    }

  } // namespace hykkt
} // namespace ReSolve
