// ANDREW TODO: cusparse (no cudss) compatibility, or at least make it work w/o preconditioner

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
    /** Constructor for MultiBasisParallelConjugateGradient without preconditioning.
     *  @param n[in] - Dimension of the system.
     *  @param num_rhs[in] - Number of copies of the systems solved at once.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     */
    MultiBasisParallelConjugateGradient::MultiBasisParallelConjugateGradient(
        index_type          n,
        index_type          num_rhs,
        MatrixHandler*      matrix_handler,
        VectorHandler*      vector_handler,
        memory::MemorySpace memspace)
      : n_(n),
        k_(num_rhs),
        matrix_handler_(matrix_handler),
        vector_handler_(vector_handler),
        memspace_(memspace)
    {
#ifdef RESOLVE_USE_CUDA
      impl_ = new MultiBasisParallelConjugateGradientCuda(vector_handler_);
#elif defined(RESOLVE_USE_HIP)
      impl_ = new MultiBasisParallelConjugateGradientHip(vector_handler_);
#endif

      if (k_ != 1 && k_ != 2 && k_ != 4 && k_ != 8)
      {
        out::warning() << "num_rhs must be 1, 2, 4, or 8!";
      }
    }

    /** Constructor for MultiBasisParallelConjugateGradient with preconditioning.
     *  @param n[in] - Dimension of the system.
     *  @param preconditioner[in] - Factorization of the preconditioner.
     *  @param num_rhs[in] - Number of copies of the systems solved at once.
     *  @param matrix_handler[in] - Matrix handler for the selected backend.
     *  @param vector_handler[in] - Vector handler for the selected backend.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     */
    MultiBasisParallelConjugateGradient::MultiBasisParallelConjugateGradient(
        index_type          n,
        index_type          num_rhs,
        Preconditioner*     preconditioner,
        MatrixHandler*      matrix_handler,
        VectorHandler*      vector_handler,
        memory::MemorySpace memspace)
      : n_(n),
        k_(num_rhs),
        preconditioner_(preconditioner),
        matrix_handler_(matrix_handler),
        vector_handler_(vector_handler),
        memspace_(memspace)
    {
#ifdef RESOLVE_USE_CUDA
      impl_ = new MultiBasisParallelConjugateGradientCuda(vector_handler_);
#elif defined(RESOLVE_USE_HIP)
      impl_ = new MultiBasisParallelConjugateGradientHip(vector_handler_);
#endif

      if (k_ != 1 && k_ != 2 && k_ != 4 && k_ != 8)
      {
        out::warning() << "num_rhs must be 1, 2, 4, or 8!";
      }

      if (preconditioner_)
      {
        enable_preconditioning_ = true;
      }
    }

    MultiBasisParallelConjugateGradient::~MultiBasisParallelConjugateGradient()
    {
      delete X_scal_0_;
      delete X_res_;
      delete B_res_;
      delete B_;
      delete R_;
      delete P_;
      delete Xi_inv_;
      delete Delta_;
      delete Psi_;
      delete Temp_nxk_;
      delete Temp_kxk_;
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
      if (enable_preconditioning_)
      {
        delete Z_;
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
    
    /**
     * @brief Reloads pointer to the preconditioner. If a preconditioner is not previously set, and 
     * the preconditioner argument is not a nullptr, this will enable preconditioning.
     * @param[in] preconditioner - Factorization of the preconditioner
     */
    void MultiBasisParallelConjugateGradient::addPreconditionerInfo(Preconditioner* preconditioner)
    {
      preconditioner_ = preconditioner;
      if (preconditioner)
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
      P_ = new vector::Vector(n_, k_);
      Xi_inv_ = new vector::Vector(k_, k_);
      Delta_ = new vector::Vector(k_, k_);
      Psi_ = new vector::Vector(k_, k_);
      Temp_nxk_ = new vector::Vector(n_, k_);
      Temp_kxk_ = new vector::Vector(k_, k_);
      c_ = new vector::Vector(k_, 1);
      r_ = new vector::Vector(n_, 1);

      X_scal_0_->allocate(memspace_);
      X_res_->allocate(memspace_);
      B_res_->allocate(memspace_);
      B_->allocate(memspace_);
      R_->allocate(memspace_);
      P_->allocate(memspace_);
      Xi_inv_->allocate(memspace_);
      Delta_->allocate(memspace_);
      Psi_->allocate(memspace_);
      Temp_nxk_->allocate(memspace_);
      Temp_kxk_->allocate(memspace_);
      c_->allocate(memspace_);
      r_->allocate(memspace_);
      
      X_res_->setToZero(memspace_);
      Psi_->setToZero(memspace_);
      Temp_kxk_->setToZero(memspace_);
      c_->setToZero(memspace_);
      r_->setToZero(memspace_);

      if (!enable_diagonal_scaling_)
      {
        A_scal_ = A_;
        b_scal_ = b_;
        R_scal_ = R_;
      }

      if (enable_preconditioning_)
      {
        Z_ = new vector::Vector(n_, k_);
        Z_->allocate(memspace_);
      }
      else
      {
        Z_ = R_scal_;
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

      if (enable_preconditioning_)
      {
        preconditioner_->apply(R_scal_, Z_);
      }

      // P, Psi = qr(R)
      P_->copyFromExternal(Z_, memspace_, memspace_); // with preconditioner, this is L_inv * R_res
      impl_->qr(P_, Psi_, memspace_);
      
      // deviceSynchronize(); // timing
      auto iterative_start = std::chrono::steady_clock::now();
      std::chrono::time_point<std::chrono::steady_clock> iterative_end;

      int i;
      for (i = 0; i < itmax_; i++)
      {
        // deviceSynchronize(); // timing
        // deviceSynchronize(); // timing
        // auto it_start = std::chrono::steady_clock::now(); // timing
        
        // SpMM
        // deviceSynchronize(); // timing
        // auto spmm_start = std::chrono::steady_clock::now(); // timing
        // matrix_handler_->matvec(A_scal_, P_, Temp_nxk_, &ONE, &ZERO, memspace_);
        impl_->SpMM(A_scal_, P_, Temp_nxk_);

        // deviceSynchronize(); // timing
        // auto spmm_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] spmm: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(spmm_end - spmm_start).count()); // timing
        
        // 2. S^T * (A * S), multTSMTTSMSymmetric
        // auto gemm_xi_start = std::chrono::steady_clock::now(); // timing
        // real_type a = vector_handler_->dot(P_, Temp_nxk_, memspace_);
        // vector_handler_->gemm('T', 'N', ONE, ZERO, P_, Temp_nxk_, Xi_inv_, memspace_);
        impl_->multTSMTTSMSymmetric(P_, Temp_nxk_, Xi_inv_, memspace_);
        // deviceSynchronize(); // timing
        // auto gemm_xi_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] gemm_xi: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(gemm_xi_end - gemm_xi_start).count()); // timing

        // Sigma = P^T * R_scal (Temp_kxk_ = Sigma)
        // deviceSynchronize(); // timing
        // auto sigma_start = std::chrono::steady_clock::now(); // timing
        impl_->multTSMTTSMAsymmetric(P_, R_scal_, Temp_kxk_, memspace_); // ANDREW TODO: Z_ here? instead of R_scal_
        // deviceSynchronize(); // timing
        // auto sigma_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] sigma: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(sigma_end - sigma_start).count()); // timing

        // Cholesky, X_res, R_scal
        // deviceSynchronize(); // timing
        // auto chol_update_start = std::chrono::steady_clock::now(); // timing
        // if (vector_handler_->choleskyFactorize(Xi_inv_, 'D', memspace_) != 0)
        // {
        //   out::error() << "Cholesky failed!";
        //   return 1;
        // }
        // Temp_kxk_->copyFromExternal(DELETE, memspace_, memspace_);
        // vector_handler_->choleskyFactorizeSolve(Xi_inv_->getData(memspace_), Temp_kxk_, 'D', memspace_); // Temp_kxk = Xi * Sigma
        // vector_handler_->gemm('N', 'N', ONE, ONE, P_, Temp_kxk_, X_res_, memspace_);
        // vector_handler_->gemm('N', 'N', MINUS_ONE, ONE, Temp_nxk_, Temp_kxk_, R_scal_, memspace_);
        impl_->choleskyFactorizeSolve(Xi_inv_, Temp_kxk_, Temp_kxk_);
        impl_->updateXR(P_, Temp_nxk_, Temp_kxk_, X_res_, R_scal_);
        // deviceSynchronize(); // timing
        // auto chol_update_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] cholesky & update: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(chol_update_end - chol_update_start).count()); // timing

        // deviceSynchronize(); // timing
        // auto preconditioning_start = std::chrono::steady_clock::now(); // timing
        if (enable_preconditioning_)
        { // ANDREW TODO: maybe move this after error check & spli
          preconditioner_->apply(R_scal_, Z_);
        }
        // deviceSynchronize(); // timing
        // auto preconditioning_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] preconditioning: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(preconditioning_end - preconditioning_start).count()); // timing

        // deviceSynchronize(); // timing
        // auto delta_update_start = std::chrono::steady_clock::now(); // timing
        impl_->multTSMTTSMAsymmetric(Temp_nxk_, Z_, Delta_, memspace_);
        // deviceSynchronize(); // timing
        // auto delta_update_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] delta: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(delta_update_end - delta_update_start).count()); // timing

        if (enable_diagonal_scaling_)
        {
          // deviceSynchronize(); // timing
          // auto r_scale_start = std::chrono::steady_clock::now(); // timing
          R_->copyFromExternal(R_scal_, memspace_, memspace_);
          vector_handler_->scal(d_, R_, memspace_);
          vector_handler_->scal(b_norm_, R_, memspace_);
          // deviceSynchronize(); // timing
          // auto r_scale_end = std::chrono::steady_clock::now(); // timing
          // printf("  [Iteration %d] R scale: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(r_scale_end - r_scale_start).count()); // timing

        }

        // Best basis
        // deviceSynchronize(); // timing
        // auto basis_loop_start = std::chrono::steady_clock::now(); // timing
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

        best_basis_error = best_basis_r_norm / b_norm_;
        // auto basis_loop_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] best basis: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(basis_loop_end - basis_loop_start).count()); // timing
        // printf("error %f\n", best_basis_error);

        // std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << best_basis_error << '\n';
        // if (k_ == 1) printf("%.10e\n", best_basis_error);
        // if (best_basis_error > 1e10)
        // {
        //   printf("hi\n");
        // }
        // 7. Convergence Checking & Final Calculations Block
        if (best_basis_error < initial_tol_)
        {
          bool use_best_basis = false;
          real_type r_norm;

          // auto conv_block_start = std::chrono::steady_clock::now(); // timing

          // if (true)
          if (k_ == 1)
          {
            lincomb_error = best_basis_error;
            r_norm = best_basis_r_norm;
          }
          else
          {
            // deviceSynchronize(); // timing
            // A * X = B - R
            Temp_nxk_->copyFromExternal(B_, memspace_, memspace_);
            vector_handler_->axpy(MINUS_ONE, R_, Temp_nxk_, memspace_);

            // (AX)^T * AX * c = (AX)^T * b
            // vector_handler_->gemm('T', 'N', ONE, ZERO, Temp_nxk_, Temp_nxk_, Temp_kxk_, memspace_);
            impl_->multTSMTTSMSymmetric(Temp_nxk_, Temp_nxk_, Temp_kxk_, memspace_); // use innerProductTSM
            vector_handler_->gemv('T', k_, ONE, ZERO, Temp_nxk_, b_, c_, memspace_);
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
              use_best_basis = true;
            }
            else{
              lincomb_error = r_norm / b_norm_;
              // printf("%.5e\n", lincomb_error);
            }
          }
          
          if (lincomb_error < convergence_tol_)
          {
            vector_handler_->geam('N', 'N', ONE, ONE, X_res_, X_scal_0_, Temp_nxk_, memspace_);
            if (enable_diagonal_scaling_)
            {
              vector_handler_->scal(d_inv_, Temp_nxk_, memspace_);
              vector_handler_->scal(b_norm_, Temp_nxk_, memspace_);
            }

            if (use_best_basis)
            {
              x_->copyFromExternal(Temp_nxk_->getData(best_basis, memspace_), memspace_, memspace_);
            }
            else
            {
              vector_handler_->gemm('N', 'N', ONE, ZERO, Temp_nxk_, c_, x_, memspace_);
            }
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = (end - start);

            deviceSynchronize();
            // auto conv_block_end = std::chrono::steady_clock::now(); // timing
            // printf("  [Iteration %d] convergence check: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(conv_block_end - conv_block_start).count()); // timing

            printf("MBPCG convergence occured at iteration %d. Total Solve Time: %f ms\n", i, elapsed.count());
            printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / i);
            printf("Error: %.5e, best basis' error: %.5e\n", lincomb_error, best_basis_error);
            return 0;
          }
          // deviceSynchronize(); // timing
          // auto conv_block_end = std::chrono::steady_clock::now(); // timing
          // printf("  [Iteration %d] convergence_overhead (failed conv_tol): %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(conv_block_end - conv_block_start).count()); // timing
        }

        // Update P
        // deviceSynchronize(); // timing
        // auto p_update_start = std::chrono::steady_clock::now(); // timing
        // vector_handler_->choleskyFactorizeSolve(Xi_inv_->getData(memspace_), Temp_nxk_, 'R', memspace_); // Temp_nxk_ now contains A * S * Xi
        // vector_handler_->axpy(MINUS_ONE, Temp_nxk_, Delta_, memspace_);
        impl_->updateP(P_, Z_, Xi_inv_, Delta_, memspace_);
        // deviceSynchronize(); // timing
        // auto p_update_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] P update: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(p_update_end - p_update_start).count()); // timing

        // QR
        // deviceSynchronize(); // timing
        // auto qr_start = std::chrono::steady_clock::now(); // timing
        Psi_->setToZero(memspace_);
        impl_->qr(P_, Psi_, memspace_);
        // deviceSynchronize(); // timing
        // auto qr_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] QR: %f ms\n", i, static_cast<std::chrono::duration<double, std::milli>>(qr_end - qr_start).count()); // timing

        // auto it_end = std::chrono::steady_clock::now(); // timing
        // printf("  [Iteration %d] Total: %f ms\n\n", i, static_cast<std::chrono::duration<double, std::milli>>(it_end - it_start).count()); // timing
        iterative_end = std::chrono::steady_clock::now();
      }

      if (i == itmax_)
      {
        deviceSynchronize();
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("No MBPCG convergence in %d iterations\n", itmax_);
        printf("Total Solve Time: %.10f ms, best basis' error: %.5e\n", elapsed.count(), best_basis_error);
        printf("Per iteration time: %.10f\n", (std::chrono::duration<double, std::milli>(iterative_end - iterative_start)).count() / i);
        return 1;
      }

      return 0;
    }
  } // namespace hykkt
} // namespace ReSolve
