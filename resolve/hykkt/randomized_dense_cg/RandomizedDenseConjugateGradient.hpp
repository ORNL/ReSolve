/**
 * @file RandomizedDenseConjugateGradient.hpp
 * @brief Schur complement conjugate gradient solver for HyKKT.
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

#include <random>

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    class RandomizedDenseConjugateGradient
    {
    public:
      /**
       * @brief Constructor for RandomizedDenseConjugateGradient.
       *
       * The solver uses caller-provided matrix and vector handlers so the same solver can be run with CPU, CUDA, or HIP backends.
       *
       * @param[in] n Dimension of outer system.
       * @param[in] choleskySolver Factorization of Hgamma to use for direct solves.
       * @param[in] matrix_handler Matrix handler for the selected backend.
       * @param[in] vector_handler Vector handler for the selected backend.
       * @param[in] memspace Memory space of incoming data and for computation.
       */
      RandomizedDenseConjugateGradient(index_type          n,
                                  index_type          k,
                                  CholeskySolver*     cholesky_solver,
                                  MatrixHandler*      matrix_handler,
                                  VectorHandler*      vector_handler,
                                  memory::MemorySpace memspace);
      ~RandomizedDenseConjugateGradient();

      void addMatrixInfo(vector::Vector* A);
      void addVectorInfo(vector::Vector* x_0, vector::Vector* b);
      void addPreconditionerInfo(matrix::Csr* L);
      void setSolverTolerance(double initial_tol, double convergence_tol);
      void setSolverItmax(int itmax);

      void setup();
      void precondition();
      void randomVector(vector::Vector* v, real_type min, real_type max);
      void generateGuesses();
      void computeQr(vector::Vector* V, vector::Vector* H);
      int  solve();

    private:
      index_type n_;             // Dimension of outer system
      index_type k_;             // 
      int        itmax_ = 3000;   // Maximum iterations for conjugate gradient
      real_type  initial_tol_   = 1e-12; // Solver tolerance for Schur
      real_type  convergence_tol_   = 1e-12; // Solver tolerance for Schur
    
      MatrixHandler* matrix_handler_{nullptr}; ///< Backend-specific matrix handler.
      VectorHandler* vector_handler_{nullptr}; ///< Backend-specific vector handler.
      
      CholeskySolver* cholesky_solver_{nullptr};
      GramSchmidt gram_schmidt_;

      vector::Vector* A_{nullptr};
      vector::Vector* x_{nullptr};   // LHS of entire system
      vector::Vector* b_{nullptr};   // RHS of entire system

      real_type A_norm_ = 0.0;
      real_type b_norm_ = 0.0;
      
      // Matrices used for conjugate gradient
      vector::Vector* A_prec_{nullptr};
      vector::Vector* A_prec_tr_{nullptr};
      matrix::Csr* L_{nullptr};

      // Vectors used for conjugate gradient
      vector::Vector* X_prec_0_{nullptr};
      vector::Vector* X_res_{nullptr};
      vector::Vector* b_prec_{nullptr};   // RHS of entire system
      vector::Vector* B_res_{nullptr};
      vector::Vector* B_{nullptr};
      vector::Vector* R_{nullptr};
      vector::Vector* R_prec_{nullptr};
      vector::Vector* S_{nullptr};
      vector::Vector* Xi_inv_{nullptr};
      vector::Vector* W_{nullptr};
      vector::Vector* Sigma_{nullptr};
      vector::Vector* Zeta_{nullptr};
      vector::Vector* Temp_nxk_{nullptr};
      vector::Vector* Temp_nxk1_{nullptr};
      vector::Vector* Temp_kxk_{nullptr};
      vector::Vector* A_S_{nullptr};
      vector::Vector* c_{nullptr};
      vector::Vector* r_{nullptr};

      std::mt19937 generator_;

      memory::MemorySpace memspace_;
    }; // class RandomizedDenseConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
