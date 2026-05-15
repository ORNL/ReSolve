/**
 * @file SchurComplementConjugateGradient.hpp
 * @brief Currently CPU-only.
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  namespace hykkt
  {
    class SchurComplementConjugateGradient
    {
    public:
      SchurComplementConjugateGradient(index_type n, index_type m, CholeskySolver* choleskySolver, memory::MemorySpace memspace);

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

#ifdef RESOLVE_USE_CUDA
      LinAlgWorkspaceCUDA workspace_;
#elif RESOLVE_USE_HIP
      LinAlgWorkspaceHIP workspace_;
#else
      LinAlgWorkspaceCpu workspace_;
#endif
      MatrixHandler matrixhandler_;
      VectorHandler vectorhandler_;

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

      MemoryHandler       mem_;
      memory::MemorySpace memspace_;
    }; // class SchurComplementConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
