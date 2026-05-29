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
       * @param[in] memspace Memory space of incoming data and for computation.
       * @param[in] matrix_handler Matrix handler for the selected backend.
       * @param[in] vector_handler Vector handler for the selected backend.
       */
      SchurComplementConjugateGradient(index_type n, index_type m, CholeskySolver* choleskySolver, memory::MemorySpace memspace, MatrixHandler& matrix_handler, VectorHandler& vector_handler);

      void addMatrixInfo(matrix::Csr* jc, matrix::Csr* jc_tr);
      void addVectorInfo(vector::Vector* x0, vector::Vector* b);
      void updateCholeskySolver(CholeskySolver* choleskySolver);
      void setSolverTolerance(double tol);
      void setSolverItmax(int itmax);

      void setup();
      int  solve();

    private:
      index_type n_;             // Dimension of outer system
      index_type m_;             // Dimension of inner system
      int        itmax_ = 100;   // Maximum iterations for conjugate gradient
      double     tol_   = 1e-12; // Solver tolerance for Schur

      MatrixHandler& matrix_handler_; ///< Backend-specific matrix handler.
      VectorHandler& vector_handler_; ///< Backend-specific vector handler.

      CholeskySolver* choleskySolver_; // Cholesky factorization on 1,1 block

      matrix::Csr* jc_;
      matrix::Csr* jc_tr_;

      vector::Vector* x0_; // LHS of entire system
      vector::Vector* b_;  // RHS of entire system

      // scalars used for conjugate gradient
      double beta_;
      double delta_;
      double alpha_;
      double minalpha_;
      double gam_i_;
      double gam_i1_;

      // Vectors used for conjugate gradient
      vector::Vector y_; // Internal RHS of system
      vector::Vector z_; // Internal LHS of system
      vector::Vector r_; // Residual
      vector::Vector p_;
      vector::Vector s_;
      vector::Vector w_;

      memory::MemorySpace memspace_;
    }; // class SchurComplementConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
