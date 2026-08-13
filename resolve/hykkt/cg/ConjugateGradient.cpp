#include "ConjugateGradient.hpp"

#include <cmath>
#include <chrono>

#include <resolve/Common.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include <hip/hip_runtime.h>
namespace ReSolve
{
  using out = io::Logger;

  namespace hykkt
  {
    /** Constructor for ConjugateGradient without preconditioning.
     *  @param n[in] - Dimension of the system.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     *  @param memspace[in] - Memory space of incoming data and for computation.
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
    }

    /** Constructor for ConjugateGradient with preconditioning.
     *  @param n[in] - Dimension of the system.
     *  @param cholesky_solver[in] - Factorization of the preconditioner.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     */
    ConjugateGradient::ConjugateGradient(
        index_type          n,
        CholeskySolver*     cholesky_solver,
        MatrixHandler*      matrix_handler,
        VectorHandler*      vector_handler,
        memory::MemorySpace memspace)
      : n_(n),
        cholesky_solver_(cholesky_solver),
        matrix_handler_(matrix_handler),
        vector_handler_(vector_handler),
        memspace_(memspace)
    {
      if (cholesky_solver_)
      {
        enable_preconditioning_ = true;
      }
    }

    ConjugateGradient::~ConjugateGradient()
    {
      delete r_;
      delete p_;
      delete s_;
      delete w_;
      if (enable_diagonal_scaling_)
      {
        delete d_;
        delete d_inv_;
        delete A_scal_;
        delete b_scal_;
        delete r_scal_;
      }
      if (enable_preconditioning_)
      {
        delete z_;
      }
    }

    /**
     * @brief Loads or reloads matrix pointers to the solver
     * @param[in] A - Pointer to the A matrix in CSR format.
     */
    void ConjugateGradient::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
    }

    /**
     * @brief Loads or reloads vector pointers to the solver
     * @param[in] x - Pointer to the left-hand side vector. It should contain the initial guess vector.
     * @param[in] b - Pointer to the right-hand side vector.
     */
    void ConjugateGradient::addVectorInfo(vector::Vector* x, vector::Vector* b)
    {
      x_ = x;
      b_   = b;
    }
    
    /**
     * @brief Reloads pointer to the Cholesky solver for preconditioning. If a Cholesky solver is
     * not previously set, and cholesky_solver is not a nullptr, this will enable preconditioning.
     * @param[in] cholesky_solver - Factorization of the preconditioner
     */
    void ConjugateGradient::updateCholeskySolver(CholeskySolver* cholesky_solver)
    {
      cholesky_solver_ = cholesky_solver;
      if (cholesky_solver)
      {
        if (enable_preconditioning_)
        {
          out::warning() << "Avoid combining preconditioning and diagonal scaling. The diagonal scaling will be done in reference to the original matrix, not the preconditioned matrix.";
        }
        enable_preconditioning_ = true;
      }
      else
      {
        enable_preconditioning_ = false;
      }
    }

    void ConjugateGradient::setSolverTolerance(double tol)
    {
      tol_ = tol;
    }

    void ConjugateGradient::setSolverItmax(int itmax)
    {
      itmax_ = itmax;
    }

    /*
    * // ...
    * For repeated solves, this only needs to be called once
    */
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

      // Clear out any nan's
      p_->setToZero(memspace_);
      s_->setToZero(memspace_);
      w_->setToZero(memspace_);

      if (!enable_diagonal_scaling_)
      {
        A_scal_ = A_;
        b_scal_ = b_;
        r_scal_ = r_;
      }

      if (enable_preconditioning_)
      {
        z_ = new vector::Vector(n_);
        z_->allocate(memspace_);
      }
      else
      {
        z_ = r_scal_;
      }

      beta_ = 0;
    }

    /** // ... ANDREW TODO: make this interface better? like a boolean toggle?
     * // ANDREW TODO again: scale b by b_norm_ regardless of diagonal scaling flag?
    * @post b_scal_, r_scal_ ...
    */
    void ConjugateGradient::diagonalScale()
    {
      using namespace constants;

      if (enable_preconditioning_)
      {
        out::warning() << "Avoid combining preconditioning and diagonal scaling. The diagonal scaling will be done in reference to the original matrix, not the preconditioned matrix.";
      }

      d_ = new vector::Vector(n_);
      d_->allocate(memspace_);
      matrix_handler_->extractRootDiagonal(A_, d_, memspace_);

      d_inv_ = new vector::Vector(n_);
      d_inv_->allocate(memspace_);
      vector_handler_->elementWiseInverse(d_, d_inv_, memspace_);
      
      A_scal_ = new matrix::Csr(n_, n_, A_->getNnz());
      b_scal_ = new vector::Vector(n_);
      r_scal_ = new vector::Vector(n_);
      A_scal_->allocateMatrixData(memspace_);
      b_scal_->allocate(memspace_);
      r_scal_->allocate(memspace_);

      // A_scal_ = L^-1 * A * L^-T
      // variable names are a bit messed up
      A_scal_->copyFromExternal(A_->getRowData(memspace_),
                                A_->getColData(memspace_),
                                A_->getValues(memspace_),
                                memspace_,
                                memspace_);
      matrix_handler_->leftScale(d_inv_, A_scal_, memspace_);
      matrix_handler_->rightScale(A_scal_, d_inv_, memspace_);

      enable_diagonal_scaling_ = true;
    }

    int ConjugateGradient::solve()
    {
      using namespace constants;

      auto start = std::chrono::steady_clock::now();

      x_->setToZero(memspace_);

      b_norm_ = std::sqrt(vector_handler_->dot(b_, b_, memspace_));
      if (enable_diagonal_scaling_)
      {
        // b_scal_ = 1 / b_norm * L^-1 * b
        b_scal_->copyFromExternal(b_, memspace_, memspace_);
        vector_handler_->scal(d_inv_, b_scal_, memspace_);
        vector_handler_->scal(1.0 / b_norm_, b_scal_, memspace_);
      }
      r_scal_->copyFromExternal(b_scal_, memspace_, memspace_);

      matrix_handler_->matvec(A_scal_, x_, r_scal_, &MINUS_ONE, &ONE, memspace_);
      if (enable_preconditioning_)
      {
        cholesky_solver_->solve(z_, r_scal_);
      }
      gamma_i_ = vector_handler_->dot(r_scal_, z_, memspace_);

      matrix_handler_->matvec(A_scal_, z_, w_, &ONE, &ZERO, memspace_);
      delta_ = vector_handler_->dot(w_, z_, memspace_);
      alpha_ = gamma_i_ / delta_;

      auto iterative_start = std::chrono::steady_clock::now();
      std::chrono::time_point<std::chrono::steady_clock> iterative_end;

      int i;
      for (i = 0; i < itmax_; i++)
      {
      auto start = std::chrono::steady_clock::now();
        vector_handler_->scal(beta_, p_, memspace_);
        vector_handler_->axpy(ONE, z_, p_, memspace_);
        vector_handler_->scal(beta_, s_, memspace_);
        vector_handler_->axpy(ONE, w_, s_, memspace_);
        vector_handler_->axpy(alpha_, p_, x_, memspace_);
        vector_handler_->axpy(-alpha_, s_, r_scal_, memspace_);

        if (enable_diagonal_scaling_)
        {
          r_->copyFromExternal(r_scal_, memspace_, memspace_);
          vector_handler_->scal(d_, r_, memspace_);
          vector_handler_->scal(b_norm_, r_, memspace_); // can maybe save one operation in computing error
        }
        r_norm_ = std::sqrt(vector_handler_->dot(r_, r_, memspace_));
        error_ = r_norm_ / b_norm_;
        if (error_ < tol_)
        {
          auto end = std::chrono::steady_clock::now();
          std::chrono::duration<double, std::milli> elapsed = (end - start);
          printf("Convergence occured at iteration %d. Took %f ms.\n", i, elapsed.count());
          printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / i);
          break;
        }
        if (enable_preconditioning_)
        {
          cholesky_solver_->solve(z_, r_scal_);
        }
        matrix_handler_->matvec(A_scal_, z_, w_, &ONE, &ZERO, memspace_);
        delta_   = vector_handler_->dot(w_, z_, memspace_);
        gamma_i1_ = vector_handler_->dot(r_scal_, z_, memspace_);
        beta_    = gamma_i1_ / gamma_i_;
        gamma_i_ = gamma_i1_;
        alpha_   = gamma_i_ / (delta_ - beta_ * gamma_i_ / alpha_);
        // auto end = std::chrono::steady_clock::now();
        // std::chrono::duration<double, std::milli> elapsed = (end - start);
        // printf("time = %f, error = %f\n", elapsed.count(), error_);
        iterative_end = std::chrono::steady_clock::now();
      }

      printf("Conjugate gradient error is %32.32g \n", error_);
      if (i == itmax_)
      {
        printf("No CG convergence in %d iterations\n", itmax_);
        printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / i);
        return 1;
      }
      return 0;
    }

  } // namespace hykkt
} // namespace ReSolve
