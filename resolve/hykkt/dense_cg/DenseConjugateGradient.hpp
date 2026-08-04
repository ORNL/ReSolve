/**
 * @file DenseConjugateGradient.hpp
 * @brief Schur complement conjugate gradient solver for HyKKT.
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/hykkt/randomized_cg/RandomizedConjugateGradientImpl.hpp>
#ifdef RESOLVE_USE_CUDA
#include <resolve/hykkt/randomized_cg/RandomizedConjugateGradientCuda.hpp>
#elif defined(RESOLVE_USE_HIP)
#include <resolve/hykkt/randomized_cg/RandomizedConjugateGradientHip.hpp>
#endif
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    class DenseConjugateGradient
    {
    public:
      /**
       * @brief Constructor for DenseConjugateGradient.
       *
       * The solver uses caller-provided matrix and vector handlers so the same solver can be run with CPU, CUDA, or HIP backends.
       *
       * @param[in] n Dimension of outer system.
       * @param[in] m Dimension of inner system.
       * @param[in] choleskySolver Factorization of Hgamma to use for direct solves.
       * @param[in] matrix_handler Matrix handler for the selected backend.
       * @param[in] vector_handler Vector handler for the selected backend.
       * @param[in] memspace Memory space of incoming data and for computation.
       */
      DenseConjugateGradient(index_type          n,
                                       MatrixHandler*      matrix_handler_,
                                       VectorHandler*      vector_handler_,
                                       memory::MemorySpace memspace);
      ~DenseConjugateGradient();

      void addMatrixInfo(vector::Vector* A);
      void addVectorInfo(vector::Vector* x_0, vector::Vector* b);
      void addPreconditionerInfo(vector::Vector* d, vector::Vector* d_inv);
      void setSolverTolerance(double tol);
      void setSolverItmax(int itmax);

      void setup();
      void precondition();
      int  solve();

    private:
      index_type n_;             // Dimension of outer system
      int        itmax_ = 9700;   // Maximum iterations for conjugate gradient
      double     tol_   = 1e-8; // Solver tolerance for Schur

      MatrixHandler* matrix_handler_{nullptr}; ///< Backend-specific matrix handler.
      VectorHandler* vector_handler_{nullptr}; ///< Backend-specific vector handler.

      RandomizedConjugateGradientImpl* impl_{nullptr};

      vector::Vector* A_{nullptr};

      vector::Vector* x_0_{nullptr}; // LHS of entire system
      vector::Vector* b_{nullptr};   // RHS of entire system

      vector::Vector* A_prec_{nullptr};
      vector::Vector* d_{nullptr};
      vector::Vector* d_inv_{nullptr};

      // scalars used for conjugate gradient
      double beta_;
      double delta_;
      double alpha_;
      double gamma_i_;
      double gamma_i1_;
      double b_norm_;
      double r_norm_;
      double error_;

      // Vectors used for conjugate gradient
      vector::Vector* r_{nullptr}; // Residual
      vector::Vector* r_prec_{nullptr};
      vector::Vector* b_prec_{nullptr};
      vector::Vector* p_{nullptr};
      vector::Vector* s_{nullptr};
      vector::Vector* w_{nullptr};

      memory::MemorySpace memspace_;
    }; // class DenseConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
