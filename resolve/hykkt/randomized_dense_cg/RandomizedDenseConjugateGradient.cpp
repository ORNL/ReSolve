#include "RandomizedDenseConjugateGradient.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>

#include <resolve/Common.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
#include <resolve/utilities/logger/Logger.hpp>

// #include <cuda_runtime.h>

namespace ReSolve
{
  using out = io::Logger;

  namespace hykkt
  {
    /** Constructor for RandomizedDenseConjugateGradient.
     *  @param n[in] - Dimension of outer system.
     *  @param m[in] - Dimension of inner system.
     *  @param choleskySolver[in] - Factorization of H_gamma to use for direct solve.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     */
    RandomizedDenseConjugateGradient::RandomizedDenseConjugateGradient(
        index_type          n,
        index_type          k,
        CholeskySolver*     cholesky_solver,
        MatrixHandler*      matrix_handler,
        VectorHandler*      vector_handler,
        memory::MemorySpace memspace)
      : n_(n),
        k_(k),
        cholesky_solver_(cholesky_solver),
        matrix_handler_(matrix_handler),
        vector_handler_(vector_handler),
        memspace_(memspace),
        gram_schmidt_(vector_handler_, GramSchmidt::GSVariant::CGS2),
        generator_(constants::SEED)
    {
      ;
    }

    RandomizedDenseConjugateGradient::~RandomizedDenseConjugateGradient()
    {
      delete A_prec_;
      delete A_prec_tr_;
      delete X_prec_0_;
      delete X_res_;
      delete b_prec_;
      delete B_res_;
      delete B_;
      delete R_;
      delete R_prec_;
      delete S_;
      delete Xi_inv_;
      delete W_;
      delete Sigma_;
      delete Zeta_;
      delete Temp_nxk_;
      delete Temp_nxk1_;
      delete Temp_kxk_;
      delete A_S_;
      delete c_;
      delete r_;
    }

    /**
     * @brief Loads or reloads matrix pointers to the solver
     * @param[in] J - Pointer to the JC matrix in CSR format.
     */
    void RandomizedDenseConjugateGradient::addMatrixInfo(vector::Vector* A)
    {
      A_ = A;
    }

    /**
     * @brief Loads or reloads vector pointers to the solver
     * @param[in] x - Pointer to the left-hand side vector.
     * @param[in] b - Pointer to the right-hand side vector.
     */
    void RandomizedDenseConjugateGradient::addVectorInfo(vector::Vector* x, vector::Vector* b)
    {
      x_ = x;
      b_ = b;
    }

    /**
     * @brief Loads or reloads preconditioner matrix pointers to the solver. This transforms
     * the system into L^-1 * A * L^-T * y = L * b.
     * @param[in] L - Pointer to the lower triangular preconditioner matrix (L) in CSR format.
     */
    void RandomizedDenseConjugateGradient::addPreconditionerInfo(matrix::Csr* L)
    {
      L_ = L;
    }

    void RandomizedDenseConjugateGradient::setSolverTolerance(double initial_tol, double convergence_tol)
    {
      initial_tol_ = initial_tol;
      convergence_tol_ = convergence_tol;
    }

    void RandomizedDenseConjugateGradient::setSolverItmax(int itmax)
    {
      itmax_ = itmax;
    }

    void RandomizedDenseConjugateGradient::setup()
    {
      A_prec_ = new vector::Vector(n_, n_);
      A_prec_tr_ = new vector::Vector(n_, n_);
      X_prec_0_ = new vector::Vector(n_, k_);
      X_res_ = new vector::Vector(n_, k_);
      b_prec_ = new vector::Vector(n_);
      B_res_ = new vector::Vector(n_, k_);
      B_ = new vector::Vector(n_, k_);
      R_ = new vector::Vector(n_, k_);
      R_prec_ = new vector::Vector(n_, k_);
      S_ = new vector::Vector(n_, k_);
      Xi_inv_ = new vector::Vector(k_, k_);
      W_ = new vector::Vector(n_, k_);
      Sigma_ = new vector::Vector(k_, k_);
      Zeta_ = new vector::Vector(k_, k_);
      Temp_nxk_ = new vector::Vector(n_, k_);
      Temp_nxk1_ = new vector::Vector(n_, k_);
      Temp_kxk_ = new vector::Vector(k_, k_);
      A_S_ = new vector::Vector(n_, k_);
      c_ = new vector::Vector(k_);
      r_ = new vector::Vector(n_);

      A_prec_->allocate(memspace_);
      A_prec_tr_->allocate(memspace_);
      X_prec_0_->allocate(memspace_);
      X_res_->allocate(memspace_);
      b_prec_->allocate(memspace_);
      B_res_->allocate(memspace_);
      B_->allocate(memspace_);
      R_->allocate(memspace_);
      R_prec_->allocate(memspace_);
      S_->allocate(memspace_);
      Xi_inv_->allocate(memspace_);
      W_->allocate(memspace_);
      Sigma_->allocate(memspace_);
      Zeta_->allocate(memspace_);
      Temp_nxk_->allocate(memspace_);
      Temp_nxk1_->allocate(memspace_);
      Temp_kxk_->allocate(memspace_);
      A_S_->allocate(memspace_);
      c_->allocate(memspace_);
      r_->allocate(memspace_);
      
      A_norm_ = vector_handler_->norm(A_, memspace_);
      b_norm_ = vector_handler_->norm(b_, memspace_);

      gram_schmidt_.setup(n_, k_);
    }

    void RandomizedDenseConjugateGradient::precondition()
    {
      using namespace constants;

      // A_prec = L^-1 * A * L^-T
      // variable names are a bit messed up
      cholesky_solver_->solve(A_prec_tr_, A_);
      vector_handler_->geam('T', 'N', ONE, ZERO, A_prec_tr_, A_prec_tr_, A_prec_, memspace_); // operand B doesn't matter
      cholesky_solver_->solve(A_prec_tr_, A_prec_);
      vector_handler_->geam('T', 'N', ONE, ZERO, A_prec_tr_, A_prec_tr_, A_prec_, memspace_);

      // b_prec = 1 / b_norm * L^-1 * b
      cholesky_solver_->solve(b_prec_, b_);
      vector_handler_->scal(1.0 / b_norm_, b_prec_, memspace_);
    }

    // Generate starting guesses and set up residual space matrices & vectors
    void RandomizedDenseConjugateGradient::generateGuesses()
    {
      using namespace constants;

      // todo: tile add
      for (index_type i=0; i < k_; i++)
      {
        b_->copyToExternal(B_->getData(i, memspace_), memspace_, memspace_);
        b_prec_->copyToExternal(B_res_->getData(i, memspace_), memspace_, memspace_);
      }

      vector_handler_->randomVector(X_prec_0_, -1.0, 1.0, memspace_);
      vector_handler_->gemm('N', 'N', ONE, ZERO, A_prec_, X_prec_0_, Temp_nxk_, memspace_);
      real_type AX_prec_0_norm = vector_handler_->norm(Temp_nxk_, memspace_);
      real_type B_prec_norm = sqrt(static_cast<double>(k_)) * vector_handler_->norm(b_prec_, memspace_);
      real_type normalization_factor = B_prec_norm / AX_prec_0_norm;
      vector_handler_->scal(normalization_factor, X_prec_0_, memspace_);

      X_res_->setToZero(memspace_);
      vector_handler_->axpy(-normalization_factor, Temp_nxk_, B_res_, memspace_);
    }

    // todo: "X_res = X_res + P @ M" and "Tau = S * Xi" can be done in parallel
    int RandomizedDenseConjugateGradient::solve()
    {
      using namespace constants;
      
      real_type best_basis_error = std::numeric_limits<real_type>::infinity();
      real_type lincomb_error = std::numeric_limits<real_type>::infinity();

      precondition();

      std::chrono::time_point<std::chrono::steady_clock> start;
      std::chrono::time_point<std::chrono::steady_clock> end;
      generateGuesses();

      R_prec_->copyFromExternal(B_res_, memspace_, memspace_);
      
      // ADD PRECONDITIONER LATER. W = L^-1 * R
      W_->copyFromExternal(R_prec_, memspace_, memspace_); // with preconditioner, this is L_inv * R_res
      if (vector_handler_->choleskyQr(W_, Sigma_, memspace_) != 0)
      {
        printf("QR failed!\n");
        return 1;
        // todo: fallback
      }
        // printf("%f, %f, %f\n", vector_handler_->norm(R_, memspace_), vector_handler_->norm(W_, memspace_), vector_handler_->norm(Sigma_, memspace_));
        // return 0; /////////

      // S = L^-1 * W
      S_->copyFromExternal(W_, memspace_, memspace_);
      start = std::chrono::steady_clock::now();


      int i;
      for (i = 0; i < itmax_; i++)
      {        
// cudaEvent_t start, stop;
// cudaEventCreate(&start);
// cudaEventCreate(&stop);
// cudaEventRecord(start, 0);
        // Xi_inv = S^T * (A * S)
        vector_handler_->gemm('N', 'N', ONE, ZERO, A_prec_, S_, Temp_nxk_, memspace_);
        vector_handler_->gemm('T', 'N', ONE, ZERO, S_, Temp_nxk_, Xi_inv_, memspace_);

        // X_res = X_res + S * (Xi * Sigma)
        if (vector_handler_->choleskyFactorize(Xi_inv_, 'L', memspace_) != 0)
        {
          out::error() << "Cholesky failed!";
          return 1;
        }
        Temp_kxk_->copyFromExternal(Sigma_, memspace_, memspace_);
        vector_handler_->choleskySolve(Xi_inv_->getData(memspace_), Temp_kxk_, 'L', memspace_); // Temp_kxk = Xi * Sigma
        vector_handler_->gemm('N', 'N', ONE, ONE, S_, Temp_kxk_, X_res_, memspace_);
      
        // check residual. not important. implement later
        // R = R - (A * S) * Xi * Sigma (residual space). Temp_kxk_ = Xi * Sigma
        vector_handler_->gemm('N', 'N', MINUS_ONE, ONE, Temp_nxk_, Temp_kxk_, R_prec_, memspace_);

        // cudaDeviceSynchronize();
        // end = std::chrono::steady_clock::now();
        // std::chrono::duration<double, std::milli> elapsed = (end - start);
        // printf("Time: %f, error: %f\n", elapsed.count(), best_basis_error);

        // W = W - (A * S) * Xi
        vector_handler_->choleskySolve(Xi_inv_->getData(memspace_), Temp_nxk_, 'R', memspace_); // "Temp_nxk_" now contains A * S * Xi
        vector_handler_->axpy(MINUS_ONE, Temp_nxk_, W_, memspace_);
        // add preconditioner later. L is lower triangular. might need a new solver
        // TODO: reuse Xi * Sigma !!!!!!!!!!!!!!!!!!!!!!!!! POTENTIALLY REAL

        // RECOVER R (not preconditioned)
        matrix_handler_->matvec(L_, R_prec_, R_, &b_norm_, &ZERO, memspace_);

        // BEST BASIS VECTOR NORM
        index_type best_basis = -1;
        best_basis_error = std::numeric_limits<real_type>::infinity();
        real_type best_basis_r_norm = std::numeric_limits<real_type>::infinity();
        for (index_type j = 0; j < k_; j++)
        {
          real_type basis_r_norm = vector_handler_->norm(R_, j, memspace_);
          real_type basis_error = basis_r_norm / b_norm_;
          if (basis_error < best_basis_error)
          {
            best_basis = j;
            best_basis_r_norm = basis_r_norm;
            best_basis_error = basis_error;
          }
        }

        if (best_basis_error < initial_tol_)
        {
          // A * X = B - R
          Temp_nxk_->copyFromExternal(B_, memspace_, memspace_);
          vector_handler_->axpy(MINUS_ONE, R_, Temp_nxk_, memspace_);

          // (AX)^T * AX * c = (AX)^T * b
          vector_handler_->gemm('T', 'N', ONE, ZERO, Temp_nxk_, Temp_nxk_, Temp_kxk_, memspace_);
          vector_handler_->gemv('T', k_, ONE, ZERO, Temp_nxk_, b_, c_, memspace_);

          real_type x_norm;
          real_type r_norm;
          if (true)
          // if ((vector_handler_->choleskyFactorize(Temp_kxk_, 'L', memspace_) != 0) || (k_ == 1))
          {
            lincomb_error = best_basis_error;
            r_norm = best_basis_r_norm;
          }
          else
          {    
            if (vector_handler_->choleskySolve(Temp_kxk_->getData(memspace_), c_, 'L', memspace_) != 0)
            {
              printf("Cholesky solve failed!\n");
            }
              
            // r = b - AX * c, but it's actually r = AX * c - b because sign doesn't matter
            vector_handler_->gemv('N', k_, ONE, ZERO, Temp_nxk_, c_, r_, memspace_);
            x_norm = vector_handler_->norm(r_, memspace_); // later: remove this from main loop
            vector_handler_->axpy(MINUS_ONE, b_, r_, memspace_);
            r_norm = vector_handler_->norm(r_, memspace_);
            lincomb_error = r_norm / b_norm_;
          }
          // printf("Linear combination error: %.5e. Best basis error: %.5e\n", lincomb_error, best_basis_error);
          
          if (lincomb_error < convergence_tol_)
          {
            // Get ||r|| / (||A|| * ||x|| + ||b||) error    
            // R = B - AX, but actually R = AX - b
            Temp_nxk_->copyFromExternal(X_res_, memspace_, memspace_);
            vector_handler_->axpy(ONE, X_prec_0_, Temp_nxk_, memspace_);
            real_type best_basis_x_norm = vector_handler_->norm(Temp_nxk_, best_basis, memspace_);

            // Recover X
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = (end - start);
            printf("Convergence occured at iteration %d. Time: %f\n", i, elapsed.count());
            printf("||r|| / (||A|| * ||x|| + ||b||) error: %.5e, best basis error: %.5e\n",
                   r_norm / (A_norm_ * x_norm + b_norm_),
                   best_basis_r_norm / (A_norm_ * best_basis_x_norm + b_norm_));
            printf("||r|| / ||b|| error: %.5e, best basis error: %.5e\n", lincomb_error, best_basis_error);
            return 0;
          }
        }
        
        if (vector_handler_->choleskyQr(W_, Zeta_, memspace_) != 0)
        {
          out::error() << "QR failed!";
          return 1; // todo: fallback qr
        }

        // S = L_inv * W + S * Zeta.T. T is recycled as intermediate memory storage. do preconditioner later
        Temp_nxk_->copyFromExternal(W_, memspace_, memspace_);
        vector_handler_->gemm('N', 'T', ONE, ONE, S_, Zeta_, Temp_nxk_, memspace_);
        S_->copyFromExternal(Temp_nxk_, memspace_, memspace_);
        
        // Sigma = Zeta * Sigma
        vector_handler_->gemm('N', 'N', ONE, ZERO, Zeta_, Sigma_, Temp_kxk_, memspace_);
        Sigma_->copyFromExternal(Temp_kxk_, memspace_, memspace_);

// cudaEventRecord(stop, 0);
// cudaEventSynchronize(stop);
// float ms = 0;
// cudaEventElapsedTime(&ms, start, stop);
// std::cout << ms << '\n';
      }

      if (i == itmax_)
      {
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("No CG convergence in %d iterations\n", itmax_);
        printf("Time: %f, error: %.5e\n", elapsed.count(), best_basis_error);
        return 1;
      }

      return 0;
    }

  } // namespace hykkt
} // namespace ReSolve
