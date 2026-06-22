/**
 * @file RandomizedDenseConjugateGradient.hpp
 * @brief Schur complement conjugate gradient solver for HyKKT.
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/GramSchmidt.hpp>
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
                                  MatrixHandler*      matrix_handler_,
                                  VectorHandler*      vector_handler_,
                                  memory::MemorySpace memspace);
      ~RandomizedDenseConjugateGradient();

      void addMatrixInfo(vector::Vector* A);
      void addVectorInfo(vector::Vector* x_0, vector::Vector* b);
      void addPreconditionerInfo(matrix::Csr* L, matrix::Csr* L_tr);
      void setSolverTolerance(double tol);
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
      int        itmax_ = 100;   // Maximum iterations for conjugate gradient
      double     tol_   = 1e-12; // Solver tolerance for Schur
    
      MatrixHandler* matrix_handler_{nullptr}; ///< Backend-specific matrix handler.
      VectorHandler* vector_handler_{nullptr}; ///< Backend-specific vector handler.
      
      GramSchmidt gram_schmidt_;

      vector::Vector* A_{nullptr};
      vector::Vector* x_{nullptr};   // LHS of entire system
      vector::Vector* b_{nullptr};   // RHS of entire system
      
      // Matrices used for conjugate gradient
      vector::Vector* A_prec_{nullptr};
      vector::Vector* A_tr_prec_{nullptr};
      matrix::Csr* L_{nullptr};
      matrix::Csr* L_tr_{nullptr};

      // Vectors used for conjugate gradient
      vector::Vector* X_0_{nullptr};
      vector::Vector* X_res_{nullptr};
      vector::Vector* B_res_{nullptr};
      vector::Vector* R_{nullptr};
      vector::Vector* S_{nullptr};
      vector::Vector* Xi_inv_{nullptr};
      vector::Vector* W_{nullptr};
      vector::Vector* Sigma_{nullptr};
      vector::Vector* Zeta_{nullptr};
      vector::Vector* Temp_nxk_{nullptr};
      vector::Vector* Temp_kxk_{nullptr};
      vector::Vector* A_S_{nullptr};
      vector::Vector* Xi_Sigma_{nullptr};

      std::mt19937 generator_;

      memory::MemorySpace memspace_;
    }; // class RandomizedDenseConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
