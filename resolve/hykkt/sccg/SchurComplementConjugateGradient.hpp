/**
 * @file SchurComplementConjugateGradient.hpp
 * @brief Schur complement conjugate gradient solver for HyKKT.
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
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
    class SchurComplementConjugateGradient
    {
    public:
      /**
       * @brief Constructor for SchurComplementConjugateGradient.
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
      SchurComplementConjugateGradient(index_type          n,
                                       index_type          m,
                                       CholeskySolver*     choleskySolver,
                                       MatrixHandler*      matrix_handler_,
                                       VectorHandler*      vector_handler_,
                                       memory::MemorySpace memspace);
      ~SchurComplementConjugateGradient();

      void addMatrixInfo(matrix::Csr* J, matrix::Csr* J_tr);
      void addVectorInfo(vector::Vector* x_0, vector::Vector* b);
      void updateCholeskySolver(CholeskySolver* choleskySolver);
      void setSolverTolerance(double tol);
      void setSolverItmax(int itmax);

      /**
       * @brief Allocates internal work vectors.
       *
       * @pre Must be called exactly once before the first call to solve().
       */
      void setup();
      int  solve();

    private:
      index_type n_;             // Dimension of outer system
      index_type m_;             // Dimension of inner system
      int        itmax_ = 100;   // Maximum iterations for conjugate gradient
      double     tol_   = 1e-12; // Solver tolerance for Schur

      CholeskySolver* choleskySolver_{nullptr}; // Cholesky factorization on 1,1 block

      MatrixHandler* matrix_handler_{nullptr}; ///< Backend-specific matrix handler.
      VectorHandler* vector_handler_{nullptr}; ///< Backend-specific vector handler.

      matrix::Csr* J_{nullptr};
      matrix::Csr* J_tr_{nullptr};

      vector::Vector* x_0_{nullptr}; // LHS of entire system
      vector::Vector* b_{nullptr};   // RHS of entire system

      // scalars used for conjugate gradient
      double beta_;
      double delta_;
      double alpha_;
      double gamma_i_;
      double gamma_i1_;

      // Vectors used for conjugate gradient
      vector::Vector* y_{nullptr}; // Internal RHS of system
      vector::Vector* z_{nullptr}; // Internal LHS of system
      vector::Vector* r_{nullptr}; // Residual
      vector::Vector* p_{nullptr};
      vector::Vector* s_{nullptr};
      vector::Vector* w_{nullptr};

      memory::MemorySpace memspace_;
    }; // class SchurComplementConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
