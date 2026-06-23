#include "RandomizedDenseConjugateGradient.hpp"

#include <cmath>
#include <chrono>

#include <resolve/Common.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include <cuda_runtime.h>

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
        MatrixHandler*      matrix_handler,
        VectorHandler*      vector_handler,
        memory::MemorySpace memspace)
      : n_(n),
        k_(k),
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
      delete X_0_;
      delete X_res_;
      delete B_res_;
      delete R_;
      delete S_;
      delete Xi_inv_;
      delete W_;
      delete Sigma_;
      delete Zeta_;
      delete A_S_;
      delete Temp_nxk_;
      delete Temp_kxk_;
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
     * @param[in] L_tr_ - Pointer to the transpose preconditioner matrix (L^T) in CSR format.
     */
    void RandomizedDenseConjugateGradient::addPreconditionerInfo(matrix::Csr* L, matrix::Csr* L_tr)
    {
      L_ = L;
      L_tr_ = L_tr;
    }

    void RandomizedDenseConjugateGradient::setSolverTolerance(double tol)
    {
      tol_ = tol;
    }

    void RandomizedDenseConjugateGradient::setSolverItmax(int itmax)
    {
      itmax_ = itmax;
    }

    void RandomizedDenseConjugateGradient::setup()
    {
      X_0_ = new vector::Vector(n_, k_);
      X_res_ = new vector::Vector(n_, k_);
      B_res_ = new vector::Vector(n_, k_);
      R_ = new vector::Vector(n_, k_);
      S_ = new vector::Vector(n_, k_);
      Xi_inv_ = new vector::Vector(k_, k_);
      W_ = new vector::Vector(n_, k_);
      Sigma_ = new vector::Vector(k_, k_);
      Zeta_ = new vector::Vector(k_, k_);
      A_S_ = new vector::Vector(n_, k_);
      Temp_nxk_ = new vector::Vector(n_, k_);
      Temp_kxk_ = new vector::Vector(k_, k_);

      X_0_->allocateAll(memspace_);
      X_res_->allocate(memspace_);
      B_res_->allocate(memspace_);
      R_->allocate(memspace_);
      S_->allocate(memspace_);
      Xi_inv_->allocateAll(memspace_);
      W_->allocate(memspace_);
      Sigma_->allocate(memspace_);
      Zeta_->allocate(memspace_);
      A_S_->allocate(memspace_);
      Temp_nxk_->allocate(memspace_);
      Temp_kxk_->allocate(memspace_);

      gram_schmidt_.setup(n_, k_);
    }

    // Generate starting guesses and set up residual space matrices & vectors
    void RandomizedDenseConjugateGradient::generateGuesses()
    {
      using namespace constants;

      // todo: tile add
      for (index_type i=0; i < k_; i++)
      {
        b_->copyToExternal(B_res_->getData(i, memspace_), memspace_, memspace_);
      }

      vector_handler_->randomVector(X_0_, -1.0, 1.0, memspace_);
      vector_handler_->gemm('N', 'N', ONE, ZERO, A_, X_0_, Temp_nxk_, memspace_);
      real_type AX_0_norm = vector_handler_->norm(Temp_nxk_, memspace_);
      real_type B_norm = sqrt(static_cast<double>(k_)) * vector_handler_->norm(b_, memspace_);
      real_type normalization_factor = B_norm / AX_0_norm;
      vector_handler_->scal(normalization_factor, X_0_, memspace_);

      X_res_->setToZero(memspace_);
      vector_handler_->axpy(-normalization_factor, Temp_nxk_, B_res_, memspace_);
    }

    // todo: "X_res = X_res + P @ M" and "Tau = S * Xi" can be done in parallel
    int RandomizedDenseConjugateGradient::solve()
    {
      using namespace constants;

      generateGuesses();

      X_res_->setToZero(memspace_);

      R_->copyFromExternal(B_res_, memspace_, memspace_);
      
      // ADD PRECONDITIONER LATER. W = L^-1 * R
      W_->copyFromExternal(R_, memspace_, memspace_); // with preconditioner, this is L_inv * R_res
      if (k_>1)
      {
        int a = 1;
      }
      if (vector_handler_->choleskyQr(W_, Sigma_, memspace_) != 0)
      {
        printf("QR failed!\n");
        return 1;
        // todo: fallback householder qr
      }

      // S = L^-1 * W
      S_->copyFromExternal(W_, memspace_, memspace_);

      int i;
      for (i = 0; i < itmax_; i++)
      {        
// cudaEvent_t start, stop;
// cudaEventCreate(&start);
// cudaEventCreate(&stop);
// cudaEventRecord(start, 0);
        auto start = std::chrono::steady_clock::now();
        // Xi_inv = S^T * (A * S)
        vector_handler_->gemm('N', 'N', ONE, ZERO, A_, S_, Temp_nxk_, memspace_);
        vector_handler_->gemm('T', 'N', ONE, ZERO, S_, Temp_nxk_, Xi_inv_, memspace_);

        // X_res = X_res + S * (Xi * Sigma)
        vector_handler_->choleskyFactorize(Xi_inv_, 'L', memspace_);
        Temp_kxk_->copyFromExternal(Sigma_, memspace_, memspace_);
        vector_handler_->choleskySolve(Xi_inv_->getData(memspace_), Temp_kxk_, 'L', memspace_);
        vector_handler_->gemm('N', 'N', ONE, ONE, S_, Temp_kxk_, X_res_, memspace_);
      
        // check residual. not important. implement later
        // Temp_kxk_ = Xi * Sigma
        vector_handler_->gemm('N', 'N', MINUS_ONE, ONE, Temp_nxk_, Temp_kxk_, R_, memspace_);
        if (10000.0 < tol_)
        {
          printf("Convergence occured at iteration %d\n", i);
          break;
        }

        // W = W - (A * S) * Xi. Temp_nxk_ overriden
        vector_handler_->choleskySolve(Xi_inv_->getData(memspace_), Temp_nxk_, 'R', memspace_); // "Temp_nxk_" now contains A * S * Xi
        vector_handler_->axpy(MINUS_ONE, Temp_nxk_, W_, memspace_);
        // add preconditioner later. L is lower triangular. might need a new solver
        // TODO: reuse Xi * Sigma !!!!!!!!!!!!!!!!!!!!!!!!!
        
        if (vector_handler_->choleskyQr(W_, Zeta_, memspace_) != 0)
        {
          printf("QR failed!\n");
          return 1; // todo: fallback qr
        }
        // S = L_inv * W + S * Zeta.T. T is recycled as intermediate memory storage. do preconditioner later
        Temp_nxk_->copyFromExternal(W_, memspace_, memspace_);
        vector_handler_->gemm('N', 'T', ONE, ONE, S_, Zeta_, Temp_nxk_, memspace_);
        
        // Sigma = Zeta * Sigma
        vector_handler_->gemm('N', 'N', ONE, ZERO, Zeta_, Sigma_, Temp_kxk_, memspace_);
        Sigma_->copyFromExternal(Temp_kxk_, memspace_, memspace_);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = (end - start);
        printf("%f\n", elapsed.count());
        
// cudaEventRecord(stop, 0);
// cudaEventSynchronize(stop);
// float ms = 0;
// cudaEventElapsedTime(&ms, start, stop);
// std::cout << ms << '\n';
      }

      printf("Conjugate gradient error is %32.32g \n", 101010101010.1);
      if (i == itmax_)
      {
        printf("No CG convergence in %d iterations\n", itmax_);
        return 1;
      }

      return 0;
    }

  } // namespace hykkt
} // namespace ReSolve
