#include "RandomizedDenseConjugateGradient.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <limits>

#include <resolve/Common.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#ifdef RESOLVE_USE_CUDA
#include <cuda_runtime.h>
#include <resolve/hykkt/randomized_cg/RandomizedConjugateGradientCuda.hpp>
#define deviceSynchronize cudaDeviceSynchronize
#elif defined(RESOLVE_USE_HIP)
#include <hip/hip_runtime.h>
#include <resolve/hykkt/randomized_cg/RandomizedConjugateGradientHip.hpp>
#define deviceSynchronize hipDeviceSynchronize
#endif

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
      delete impl_;
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
    void RandomizedDenseConjugateGradient::addPreconditionerInfo(vector::Vector* d, vector::Vector* d_inv)
    {
      d_ = d;
      d_inv_ = d_inv;
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
#ifdef RESOLVE_USE_CUDA
      impl_ = new RandomizedConjugateGradientCuda(vector_handler_);
#elif defined(RESOLVE_USE_HIP)
      impl_ = new RandomizedConjugateGradientHip(vector_handler_);
#endif

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

      impl_->setup(k_);
    }

    void RandomizedDenseConjugateGradient::precondition()
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

      std::chrono::time_point<std::chrono::steady_clock> start;
      std::chrono::time_point<std::chrono::steady_clock> end;
      start = std::chrono::steady_clock::now();
      
      real_type best_basis_error = std::numeric_limits<real_type>::infinity();
      real_type lincomb_error = std::numeric_limits<real_type>::infinity();

      precondition();

      generateGuesses();

      R_prec_->copyFromExternal(B_res_, memspace_, memspace_);
      
      // ADD PRECONDITIONER LATER. W = L^-1 * R
      W_->copyFromExternal(R_prec_, memspace_, memspace_); // with preconditioner, this is L_inv * R_res
      Sigma_->setToZero(memspace_);
      if (impl_->choleskyQr(W_, Sigma_, memspace_) != 0)
      {
        printf("QR failed!\n");
        return 1;
        // todo: fallback
      }
        // printf("%f, %f, %f\n", vector_handler_->norm(R_, memspace_), vector_handler_->norm(W_, memspace_), vector_handler_->norm(Sigma_, memspace_));
        // return 0; /////////

      // S = L^-1 * W
      S_->copyFromExternal(W_, memspace_, memspace_);

      auto iterative_start = std::chrono::steady_clock::now();
      std::chrono::time_point<std::chrono::steady_clock> iterative_end;

      int i;
      for (i = 0; i < itmax_; i++)
      {        
        vector_handler_->gemm('N', 'N', ONE, ZERO, A_prec_, S_, Temp_nxk_, memspace_);

        // 2. S^T * (A * S) GEMM Section
        // auto gemm_xi_start = std::chrono::steady_clock::now();
        // vector_handler_->gemm('T', 'N', ONE, ZERO, S_, Temp_nxk_, Xi_inv_, memspace_);
        impl_->multTSMTTSM(S_, Temp_nxk_, Xi_inv_, memspace_);
        // deviceSynchronize(); // optional
        // auto gemm_xi_end = std::chrono::steady_clock::now();
        // printf("  [it %d] gemm_xi: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(gemm_xi_end - gemm_xi_start).count()); // optional

        // 3. Cholesky & X_res/R_prec Update Section
        // auto chol_update_start = std::chrono::steady_clock::now();
        // if (vector_handler_->choleskyFactorize(Xi_inv_, 'L', memspace_) != 0)
        // {
        //   out::error() << "Cholesky failed!";
        //   return 1;
        // }
        // Temp_kxk_->copyFromExternal(Sigma_, memspace_, memspace_);
        // vector_handler_->choleskySolve(Xi_inv_->getData(memspace_), Temp_kxk_, 'L', memspace_); // Temp_kxk = Xi * Sigma
        // vector_handler_->gemm('N', 'N', ONE, ONE, S_, Temp_kxk_, X_res_, memspace_);
        // vector_handler_->gemm('N', 'N', MINUS_ONE, ONE, Temp_nxk_, Temp_kxk_, R_prec_, memspace_);
        impl_->updateXRSplit(Xi_inv_, Sigma_, S_, Temp_nxk_, Temp_kxk_, X_res_, R_prec_);
        // impl_->updateXR(Xi_inv_, Sigma_, S_, Temp_nxk_, X_res_, R_prec_);
        // deviceSynchronize(); // optional
        // auto chol_update_end = std::chrono::steady_clock::now();
        // printf("  [it %d] cholesky & update: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(chol_update_end - chol_update_start).count()); // optional

        // 4. Update W Section (W = W - A * S * Xi)
        // auto w_update_start = std::chrono::steady_clock::now();
        // vector_handler_->choleskySolve(Xi_inv_->getData(memspace_), Temp_nxk_, 'R', memspace_); // Temp_nxk_ now contains A * S * Xi
        // vector_handler_->axpy(MINUS_ONE, Temp_nxk_, W_, memspace_);
        impl_->updateW(W_, Xi_inv_, Temp_nxk_, memspace_); // This is wrong I think. accessing garbage half of Xi_inv_?
        // deviceSynchronize(); // optional
        // auto w_update_end = std::chrono::steady_clock::now();
        // printf("  [it %d] w_update: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(w_update_end - w_update_start).count()); // optional

        // 5. Scale & Diagonal Solve R Section
        // auto r_scale_start = std::chrono::steady_clock::now();
        R_->copyFromExternal(R_prec_, memspace_, memspace_);
        vector_handler_->scal(d_, R_, memspace_);
        vector_handler_->scal(b_norm_, R_, memspace_);
        // deviceSynchronize(); // optional
        // auto r_scale_end = std::chrono::steady_clock::now();
        // printf("  [it %d] r_scale: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(r_scale_end - r_scale_start).count()); // optional

        // 6. Best Basis Vector Norm Loop Section
        // auto basis_loop_start = std::chrono::steady_clock::now();

        index_type best_basis;
        real_type best_basis_r_norm;
        impl_->bestBasis(R_, &best_basis, &best_basis_r_norm);

        // index_type best_basis = -1;
        // real_type best_basis_r_norm = std::numeric_limits<real_type>::infinity();
        // for (index_type j = 0; j < k_; j++)
        // {
        //   real_type basis_r_norm = vector_handler_->norm(R_, j, memspace_);
        //   if (basis_r_norm < best_basis_r_norm)
        //   {
        //     best_basis = j;
        //     best_basis_r_norm = basis_r_norm;
        //   }
        // }
        
        deviceSynchronize();
        best_basis_error = best_basis_r_norm / b_norm_;
        // auto basis_loop_end = std::chrono::steady_clock::now();
        // printf("  [it %d] basis_norm_loop: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(basis_loop_end - basis_loop_start).count()); // optional
        // printf("error %f\n", best_basis_error);

        // std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << best_basis_error << '\n';
        // 7. Convergence Checking & Final Calculations Block
        // if (false)
        if (best_basis_error < initial_tol_)
        {
          // auto conv_block_start = std::chrono::steady_clock::now();
          // A * X = B - R
          Temp_nxk_->copyFromExternal(B_, memspace_, memspace_);
          vector_handler_->axpy(MINUS_ONE, R_, Temp_nxk_, memspace_);

          // (AX)^T * AX * c = (AX)^T * b
          // vector_handler_->gemm('T', 'N', ONE, ZERO, Temp_nxk_, Temp_nxk_, Temp_kxk_, memspace_);
          impl_->multTSMTTSM(Temp_nxk_, Temp_nxk_, Temp_kxk_, memspace_); // use innerProductTSM
          vector_handler_->gemv('T', k_, ONE, ZERO, Temp_nxk_, b_, c_, memspace_);
          deviceSynchronize();

          real_type x_norm;
          real_type r_norm;
          // if (true)
          if (k_ == 1)
          {
            lincomb_error = best_basis_error;
            r_norm = best_basis_r_norm;
          }
          else
          {
            impl_->choleskyFactorizeSolve(Temp_kxk_, c_, c_);
            
            // r = b - AX * c
            vector_handler_->gemv('N', k_, ONE, ZERO, Temp_nxk_, c_, r_, memspace_);
            vector_handler_->axpy(MINUS_ONE, b_, r_, memspace_);
            r_norm = vector_handler_->norm(r_, memspace_);
            deviceSynchronize();
            if (r_norm > best_basis_r_norm || std::isnan(r_norm))
            {
              r_norm = best_basis_r_norm;
              lincomb_error = best_basis_error;
            }
            else{
              lincomb_error = r_norm / b_norm_;
              // printf("%.5e\n", lincomb_error);
            }
          }
          
          if (lincomb_error < convergence_tol_)
          {
            vector_handler_->geam('N', 'N', ONE, ONE, X_res_, X_prec_0_, Temp_nxk_, memspace_);
            vector_handler_->scal(d_inv_, Temp_nxk_, memspace_);
            vector_handler_->scal(b_norm_, Temp_nxk_, memspace_);

            vector_handler_->gemm('N', 'N', ONE, ZERO, Temp_nxk_, c_, x_, memspace_);
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = (end - start);

            real_type x_norm = vector_handler_->norm(x_, memspace_  );
            real_type best_basis_x_norm = vector_handler_->norm(Temp_nxk_, best_basis, memspace_);

            deviceSynchronize();
            // auto conv_block_end = std::chrono::steady_clock::now();
            // printf("  [it %d] convergence_overhead: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(conv_block_end - conv_block_start).count()); // optional

            printf("Convergence occured at iteration %d. Total Solve Time: %f ms\n", i, elapsed.count());
            printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / i);
            printf("||r|| / (||A|| * ||x|| + ||b||) error: %.5e, best basis error: %.5e\n",
                   r_norm / (A_norm_ * x_norm + b_norm_),
                   best_basis_r_norm / (A_norm_ * best_basis_x_norm + b_norm_));
            printf("||r|| / ||b|| error: %.5e, best basis error: %.5e\n", lincomb_error, best_basis_error);
            return 0;
          }
          // deviceSynchronize(); // optional
          // auto conv_block_end = std::chrono::steady_clock::now();
          // printf("  [it %d] convergence_overhead (failed conv_tol): %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(conv_block_end - conv_block_start).count()); // optional
        }

        // 8. Cholesky QR Section
        // auto qr_start = std::chrono::steady_clock::now();
        Zeta_->setToZero(memspace_);
        if (impl_->choleskyQr(W_, Zeta_, memspace_) != 0)
        {
          out::error() << "QR failed!";
          return 1;
        }
        // deviceSynchronize(); // optional
        // auto qr_end = std::chrono::steady_clock::now();
        // printf("  [it %d] qr: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(qr_end - qr_start).count()); // optional

        // 9. S and Sigma Base Orthogonalization Update Section
        // auto s_sigma_update_start = std::chrono::steady_clock::now();

        // Temp_nxk_->copyFromExternal(W_, memspace_, memspace_);
        // vector_handler_->gemm('N', 'T', ONE, ONE, S_, Zeta_, Temp_nxk_, memspace_);
        // S_->copyFromExternal(Temp_nxk_, memspace_, memspace_);
        
        // vector_handler_->gemm('N', 'N', ONE, ZERO, Zeta_, Sigma_, Temp_kxk_, memspace_);
        // Sigma_->copyFromExternal(Temp_kxk_, memspace_, memspace_);

        impl_->updateSSigma(W_, S_, Zeta_, Sigma_, memspace_);
        // deviceSynchronize(); // optional
        // auto s_sigma_update_end = std::chrono::steady_clock::now();
        // printf("  [it %d] s_sigma_update: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(s_sigma_update_end - s_sigma_update_start).count()); // optional

        // auto it_end = std::chrono::steady_clock::now();
        // printf("[Iteration %d Total]: %f ms\n\n", i, static_cast<std::chrono::duration<double, std::milli>>(it_end - it_start).count());
        iterative_end = std::chrono::steady_clock::now();
      }

      if (i == itmax_)
      {
        deviceSynchronize();
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("No CG convergence in %d iterations\n", itmax_);
        printf("Total Solve Time: %f ms, error: %.5e\n", elapsed.count(), best_basis_error);
        printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / i);
        return 1;
      }

      return 0;
    }
  } // namespace hykkt
} // namespace ReSolve
