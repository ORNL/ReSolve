#include "MultiBasisParallelConjugateGradient.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>
#include <limits>

#include <resolve/Common.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#ifdef RESOLVE_USE_CUDA
#include <cuda_runtime.h>
#define deviceSynchronize cudaDeviceSynchronize
#elif defined(RESOLVE_USE_HIP)
#include <hip/hip_runtime.h>
#define deviceSynchronize hipDeviceSynchronize
#endif

namespace ReSolve
{
  using out = io::Logger;

  namespace hykkt
  {
    /** Constructor for MultiBasisParallelConjugateGradient.
     *  @param n[in] - Dimension of the system.
     *  @param k[in] - Number of copies of the systems solved at once.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     */
    MultiBasisParallelConjugateGradient::MultiBasisParallelConjugateGradient(
        index_type          n,
        index_type          k,
        MatrixHandler*      matrix_handler,
        VectorHandler*      vector_handler,
        memory::MemorySpace memspace)
      : n_(n),
        k_(k),
        matrix_handler_(matrix_handler),
        vector_handler_(vector_handler),
        memspace_(memspace)
    {
#ifdef RESOLVE_USE_CUDA
      impl_ = new MultiBasisParallelConjugateGradientCuda(vector_handler_);
#elif defined(RESOLVE_USE_HIP)
      impl_ = new MultiBasisParallelConjugateGradientHip(vector_handler_);
#endif

      if (k != 1 && k != 2 && k != 4 && k != 8)
      {
        out::warning() << "k must be 1, 2, 4, or 8!";
      }
    }

    MultiBasisParallelConjugateGradient::~MultiBasisParallelConjugateGradient()
    {
      delete X_scal_0_;
      delete X_res_;
      delete B_res_;
      delete B_;
      delete R_;
      delete S_;
      delete Xi_inv_;
      delete W_;
      delete Sigma_;
      delete Zeta_;
      delete Temp_nxk_;
      delete Temp_kxk_;
      delete e_;
      delete c_;
      delete r_;
      if (enable_diagonal_scaling_)
      {
        delete d_;
        delete d_inv_;
        delete A_scal_;
        delete b_scal_;
        delete R_scal_;
      }

      delete impl_;
    }

    /**
     * @brief Loads or reloads matrix pointers to the solver
     * @param[in] J - Pointer to the JC matrix in CSR format.
     */
    void MultiBasisParallelConjugateGradient::addMatrixInfo(matrix::Csr* A)
    {
      A_ = A;
      A_norm_ = matrix_handler_->norm(A_, memspace_);
    }

    /**
     * @brief Loads or reloads vector pointers to the solver
     * @param[in] x - Pointer to the left-hand side vector. It should contain the initial guess vector.
     * @param[in] b - Pointer to the right-hand side vector.
     */
    void MultiBasisParallelConjugateGradient::addVectorInfo(vector::Vector* x, vector::Vector* b)
    {
      x_ = x;
      b_ = b;
    }

    void MultiBasisParallelConjugateGradient::setSolverTolerance(double initial_tol, double convergence_tol)
    {
      initial_tol_ = initial_tol;
      convergence_tol_ = convergence_tol;
    }

    void MultiBasisParallelConjugateGradient::setSolverItmax(int itmax)
    {
      itmax_ = itmax;
    }

    void MultiBasisParallelConjugateGradient::setup()
    {
      X_scal_0_ = new vector::Vector(n_, k_);
      X_res_ = new vector::Vector(n_, k_);
      B_res_ = new vector::Vector(n_, k_);
      B_ = new vector::Vector(n_, k_);
      R_ = new vector::Vector(n_, k_);
      S_ = new vector::Vector(n_, k_);
      Xi_inv_ = new vector::Vector(k_, k_);
      W_ = new vector::Vector(n_, k_);
      Sigma_ = new vector::Vector(k_, k_);
      Zeta_ = new vector::Vector(k_, k_);
      Temp_nxk_ = new vector::Vector(n_, k_);
      Temp_kxk_ = new vector::Vector(k_, k_);
      e_ = new vector::Vector(k_, 1);
      c_ = new vector::Vector(k_, 1);
      r_ = new vector::Vector(n_, 1);

      X_scal_0_->allocate(memspace_);
      X_res_->allocate(memspace_);
      B_res_->allocate(memspace_);
      B_->allocate(memspace_);
      R_->allocate(memspace_);
      S_->allocate(memspace_);
      Xi_inv_->allocate(memspace_);
      W_->allocate(memspace_);
      Sigma_->allocate(memspace_);
      Zeta_->allocate(memspace_);
      Temp_nxk_->allocate(memspace_);
      Temp_kxk_->allocate(memspace_);
      e_->allocate(memspace_);
      c_->allocate(memspace_);
      r_->allocate(memspace_);
      
      X_res_->setToZero(memspace_);
      Sigma_->setToZero(memspace_);
      Temp_kxk_->setToZero(memspace_);
      e_->setToConst(1.0, memspace_);
      c_->setToZero(memspace_);
      r_->setToZero(memspace_);

      if (!enable_diagonal_scaling_)
      {
        A_scal_ = A_;
        b_scal_ = b_;
        R_scal_ = R_;
      }

      impl_->setup(k_);
    }

    // ... called once for the matrix A
    void MultiBasisParallelConjugateGradient::diagonalScale()
    {
      using namespace constants;

      d_ = new vector::Vector(n_);
      d_->allocate(memspace_);
      matrix_handler_->extractRootDiagonal(A_, d_, memspace_);

      d_inv_ = new vector::Vector(n_);
      d_inv_->allocate(memspace_);
      vector_handler_->elementWiseInverse(d_, d_inv_, memspace_);
      
      A_scal_ = new matrix::Csr(n_, n_, A_->getNnz());
      b_scal_ = new vector::Vector(n_);
      R_scal_ = new vector::Vector(n_, k_);
      A_scal_->allocateMatrixData(memspace_);
      b_scal_->allocate(memspace_);
      R_scal_->allocate(memspace_);

      // A_scal = D^-1 * A * D^-T
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

    // Generate starting guesses and set up residual space matrices & vectors
    void MultiBasisParallelConjugateGradient::generateGuesses()
    {
      using namespace constants;

      // todo: tile add
      for (index_type i=0; i < k_; i++)
      {
        b_->copyToExternal(B_->getData(i, memspace_), memspace_, memspace_);
        b_scal_->copyToExternal(B_res_->getData(i, memspace_), memspace_, memspace_);
      }

      vector_handler_->randomVector(X_scal_0_, -1.0, 1.0, memspace_);
      // deviceSynchronize(); // for debugging
      impl_->SpMM(A_scal_, X_scal_0_, Temp_nxk_);
      // matrix_handler_->matvec(A_scal_, X_scal_0_, Temp_nxk_, &ONE, &ZERO, memspace_);
      real_type AX_scal_0_norm = vector_handler_->norm(Temp_nxk_, memspace_);
      real_type B_scal_norm = sqrt(static_cast<double>(k_)) * vector_handler_->norm(b_scal_, memspace_);
      real_type normalization_factor = B_scal_norm / AX_scal_0_norm;
      vector_handler_->scal(normalization_factor, X_scal_0_, memspace_);
      vector_handler_->axpy(-normalization_factor, Temp_nxk_, B_res_, memspace_);
    }

    int MultiBasisParallelConjugateGradient::solve()
    {
      using namespace constants;

      bool initial_tol_reached = false;
      index_type initial_tol_iter;
    
      deviceSynchronize();
      
      std::chrono::time_point<std::chrono::steady_clock> start;
      std::chrono::time_point<std::chrono::steady_clock> end;
      start = std::chrono::steady_clock::now();
    
      real_type best_basis_error = std::numeric_limits<real_type>::infinity();
      real_type lincomb_error = std::numeric_limits<real_type>::infinity();

      b_norm_ = vector_handler_->norm(b_, memspace_); // put this somewhere else? in setup()? addVectorInfo?
      if (enable_diagonal_scaling_)
      {
        // b_scal = 1 / b_norm * D^-1 * b
        b_scal_->copyFromExternal(b_, memspace_, memspace_);
        vector_handler_->scal(d_inv_, b_scal_, memspace_);
        vector_handler_->scal(1.0 / b_norm_, b_scal_, memspace_);
      }
      
      generateGuesses();
      
      R_scal_->copyFromExternal(B_res_, memspace_, memspace_);
           
      W_->copyFromExternal(R_scal_, memspace_, memspace_); // with preconditioner, this is L_inv * R_res
      impl_->choleskyQr(W_, Sigma_, memspace_);

      // S = D^-1 * W
      S_->copyFromExternal(W_, memspace_, memspace_);
      
      auto iterative_start = std::chrono::steady_clock::now();
      std::chrono::time_point<std::chrono::steady_clock> iterative_end;

      int i;
      for (i = 0; i < itmax_; i++)
      {
        // deviceSynchronize(); // optional
        // auto it_start = std::chrono::steady_clock::now();
        
        // 1. SpMM / SpMV Section
        // auto spmv_start = std::chrono::steady_clock::now();
        // matrix_handler_->matvec(A_scal_, S_, Temp_nxk_, &ONE, &ZERO, memspace_);
        impl_->SpMM(A_scal_, S_, Temp_nxk_);
        // deviceSynchronize(); // optional
        // auto spmv_end = std::chrono::steady_clock::now();
        // printf("  [it %d] spmv: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(spmv_end - spmv_start).count()); // optional

        // 2. S^T * (A * S) GEMM Section
        // auto gemm_xi_start = std::chrono::steady_clock::now();
        // vector_handler_->gemm('T', 'N', ONE, ZERO, S_, Temp_nxk_, Xi_inv_, memspace_);
        impl_->multTSMTTSMSymmetric(S_, Temp_nxk_, Xi_inv_, memspace_);
        // deviceSynchronize(); // optional
        // auto gemm_xi_end = std::chrono::steady_clock::now();
        // printf("  [it %d] gemm_xi: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(gemm_xi_end - gemm_xi_start).count()); // optional

        // 3. Cholesky & X_res/R_scal Update Section
        // auto chol_update_start = std::chrono::steady_clock::now();
        // if (vector_handler_->choleskyFactorize(Xi_inv_, 'D', memspace_) != 0)
        // {
        //   out::error() << "Cholesky failed!";
        //   return 1;
        // }
        // Temp_kxk_->copyFromExternal(Sigma_, memspace_, memspace_);
        // vector_handler_->choleskyFactorizeSolve(Xi_inv_->getData(memspace_), Temp_kxk_, 'D', memspace_); // Temp_kxk = Xi * Sigma
        // vector_handler_->gemm('N', 'N', ONE, ONE, S_, Temp_kxk_, X_res_, memspace_);
        // vector_handler_->gemm('N', 'N', MINUS_ONE, ONE, Temp_nxk_, Temp_kxk_, R_scal_, memspace_);
        impl_->updateXR(Xi_inv_, Sigma_, S_, Temp_nxk_, Temp_kxk_, X_res_, R_scal_);
        // deviceSynchronize(); // optional
        // auto chol_update_end = std::chrono::steady_clock::now();
        // printf("  [it %d] cholesky & update: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(chol_update_end - chol_update_start).count()); // optional

        // 4. Update W Section (W = W - A * S * Xi)
        // auto w_update_start = std::chrono::steady_clock::now();
        // vector_handler_->choleskyFactorizeSolve(Xi_inv_->getData(memspace_), Temp_nxk_, 'R', memspace_); // Temp_nxk_ now contains A * S * Xi
        // vector_handler_->axpy(MINUS_ONE, Temp_nxk_, W_, memspace_);
        impl_->updateW(W_, Xi_inv_, Temp_nxk_, memspace_); // This is wrong I think. accessing garbage half of Xi_inv_?
        // deviceSynchronize(); // optional
        // auto w_update_end = std::chrono::steady_clock::now();
        // printf("  [it %d] w_update: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(w_update_end - w_update_start).count()); // optional

        if (enable_diagonal_scaling_)
        {
          // 5. Scale & Diagonal Solve R Section
          // auto r_scale_start = std::chrono::steady_clock::now();
          R_->copyFromExternal(R_scal_, memspace_, memspace_);
          vector_handler_->scal(d_, R_, memspace_);
          vector_handler_->scal(b_norm_, R_, memspace_);
          // deviceSynchronize(); // optional
          // auto r_scale_end = std::chrono::steady_clock::now();
          // printf("  [it %d] r_scale: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(r_scale_end - r_scale_start).count()); // optional

        }

        if (!initial_tol_reached)
        {
          // 6. Best Basis Vector Norm Loop Section
          // auto basis_loop_start = std::chrono::steady_clock::now();
          index_type best_basis;
          real_type best_basis_r_norm;
          impl_->bestBasis(R_, &best_basis, &best_basis_r_norm);
          
          best_basis_error = best_basis_r_norm / b_norm_;
          // auto basis_loop_end = std::chrono::steady_clock::now();
          // printf("  [it %d] basis_norm_loop: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(basis_loop_end - basis_loop_start).count()); // optional
          // printf("error %f\n", best_basis_error);

          // std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << best_basis_error << '\n';
          // printf("%e\n", best_basis_error);
        
          if (best_basis_error < initial_tol_)
          {
            iterative_end = std::chrono::steady_clock::now();
            initial_tol_reached = true;
            initial_tol_iter = i;
          }
        }

        // 7. Convergence Checking & Final Calculations Block
        if (initial_tol_reached)
        {
          real_type lincomb_error;
          real_type scaling_factor;
          if (k_ == 1)
          {
            lincomb_error = vector_handler_->norm(R_, memspace_) / b_norm_;
          }
          else
          {
          // auto solution_averaging_start = std::chrono::steady_clock::now();
            Temp_nxk_->copyFromExternal(R_, memspace_, memspace_);
            Temp_kxk_->setToZero(memspace_);
            impl_->choleskyQr(Temp_nxk_, Temp_kxk_, memspace_);
            impl_->choleskySolve(Temp_kxk_, e_, c_);
            scaling_factor = 1.0 / vector_handler_->dot(e_, c_, memspace_);
            vector_handler_->gemv('N', k_, ONE, ZERO, R_, c_, r_, memspace_);
            vector_handler_->scal(scaling_factor, r_, memspace_);
            lincomb_error = vector_handler_->norm(r_, memspace_) / b_norm_;
          // deviceSynchronize();
          // auto solution_averaging_end = std::chrono::steady_clock::now();
          // printf("  [it %d] solution averaging: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(solution_averaging_end - solution_averaging_start).count()); // optional
          // printf("error %f\n", best_basis_error);
          }
          
          if (lincomb_error < convergence_tol_)
          {
            if (k_ == 1)
            {
              vector_handler_->geam('N', 'N', ONE, ONE, X_res_, X_scal_0_, x_, memspace_);
              if (enable_diagonal_scaling_)
              {
                vector_handler_->scal(d_inv_, x_, memspace_);
                vector_handler_->scal(b_norm_, x_, memspace_);
              }
            }
            else
            {
              vector_handler_->geam('N', 'N', ONE, ONE, X_res_, X_scal_0_, Temp_nxk_, memspace_);
              if (enable_diagonal_scaling_)
              {
                vector_handler_->scal(d_inv_, Temp_nxk_, memspace_);
                vector_handler_->scal(b_norm_, Temp_nxk_, memspace_);
              }
              vector_handler_->gemv('N', k_, ONE, ZERO, Temp_nxk_, c_, x_, memspace_);
              vector_handler_->scal(scaling_factor, x_, memspace_);
            }

            deviceSynchronize();
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = (end - start);

            // auto conv_block_end = std::chrono::steady_clock::now();
            // printf("  [it %d] convergence_overhead: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(conv_block_end - conv_block_start).count()); // optional

            printf("MBPCG convergence occured at iteration %d. Total Solve Time: %f ms\n", i, elapsed.count());
            printf("Initial tolerance reached at iteration %d.\n", initial_tol_iter);
            printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / (initial_tol_iter + 1));
            printf("Error: %.5e, best basis' error: %.5e\n", lincomb_error, best_basis_error);
            return 0;
          }
          // deviceSynchronize(); // optional
          // auto conv_block_end = std::chrono::steady_clock::now();
          // printf("  [it %d] convergence_overhead (failed conv_tol): %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(conv_block_end - conv_block_start).count()); // optional
        }

        // 8. Cholesky QR Section
        // auto qr_start = std::chrono::steady_clock::now();
        Zeta_->setToZero(memspace_);
        impl_->choleskyQr(W_, Zeta_, memspace_);
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
      }

      if (i == itmax_)
      {
        deviceSynchronize();
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("No MBPCG convergence in %d iterations\n", itmax_);
        printf("Total Solve Time: %.10f ms, best basis' error: %.5e\n", elapsed.count(), best_basis_error);
        printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / (i + 1));
        return 1;
      }

      return 0;
    }
  } // namespace hykkt
} // namespace ReSolve
