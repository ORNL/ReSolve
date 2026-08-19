/**
 * @file ConjugateGradient.hpp
 * @brief Schur complement conjugate gradient solver for HyKKT.
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/Preconditioner.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    class ConjugateGradient
    {
    public:
      // Set up CG without a preconditioner
      ConjugateGradient(index_type          n,
                        MatrixHandler*      matrix_handler,
                        VectorHandler*      vector_handler,
                        memory::MemorySpace memspace);
      // Set up CG with a preconditioner
      ConjugateGradient(index_type          n,
                        Preconditioner*     preconditioner,
                        MatrixHandler*      matrix_handler,
                        VectorHandler*      vector_handler,
                        memory::MemorySpace memspace);
      ~ConjugateGradient();

      void addMatrixInfo(matrix::Csr* A);
      void addVectorInfo(vector::Vector* x_0, vector::Vector* b);
      void addPreconditionerInfo(Preconditioner* preconditioner);
      void setSolverTolerance(double tol);
      void setSolverItmax(int itmax);

      void setup();
      void diagonalScale();
      int  solve();

    private:
      index_type n_;               // Dimension of outer system
      int        itmax_ = 12000;   // Maximum iterations for conjugate gradient
      double     tol_   = 1e-8;    // Solver tolerance

      MatrixHandler* matrix_handler_{nullptr}; ///< Backend-specific matrix handler.
      VectorHandler* vector_handler_{nullptr}; ///< Backend-specific vector handler.

      Preconditioner* preconditioner_{nullptr};
      
      bool enable_diagonal_scaling_{false};
      bool enable_preconditioning_{false};

      matrix::Csr* A_{nullptr};

      vector::Vector* x_{nullptr}; // LHS of entire system
      vector::Vector* b_{nullptr};   // RHS of entire system

      matrix::Csr* A_scal_{nullptr};
      vector::Vector* d_{nullptr};
      vector::Vector* d_inv_{nullptr};

      // scalars used for conjugate gradient
      real_type beta_;
      real_type delta_;
      real_type alpha_;
      real_type gamma_i_;
      real_type gamma_i1_;
      real_type b_norm_;
      real_type r_norm_;
      real_type error_;

      // Vectors used for conjugate gradient
      vector::Vector* r_{nullptr}; // Residual
      vector::Vector* r_scal_{nullptr};
      vector::Vector* b_scal_{nullptr};
      vector::Vector* p_{nullptr};
      vector::Vector* s_{nullptr};
      vector::Vector* w_{nullptr};
      vector::Vector* z_{nullptr}; // If no Cholesky solver is provided (i.e. no preconditioning), z_ will point to r_

      memory::MemorySpace memspace_;
    }; // class ConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
