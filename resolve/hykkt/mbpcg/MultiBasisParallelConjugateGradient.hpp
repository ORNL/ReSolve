/**
 * @file MultiBasisParallelConjugateGradient.hpp
 * @brief Schur complement conjugate gradient solver for HyKKT.
 */

#pragma once

#include "MultiBasisParallelConjugateGradientImpl.hpp"
#ifdef RESOLVE_USE_CUDA
#include "MultiBasisParallelConjugateGradientCuda.hpp"
#elif defined(RESOLVE_USE_HIP)
#include "MultiBasisParallelConjugateGradientHip.hpp"
#endif

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
    class MultiBasisParallelConjugateGradient
    {
    public:
      MultiBasisParallelConjugateGradient(index_type          n,
                                  index_type          k,
                                  MatrixHandler*      matrix_handler,
                                  VectorHandler*      vector_handler,
                                  memory::MemorySpace memspace);
      ~MultiBasisParallelConjugateGradient();

      void addMatrixInfo(matrix::Csr* A);
      void addVectorInfo(vector::Vector* x_0, vector::Vector* b);
      void addPreconditionerInfo(vector::Vector* d_, vector::Vector* d_inv);
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
      index_type nnz_;
      int        itmax_ = 12000;   // Maximum iterations for conjugate gradient
      real_type  initial_tol_   = 1e-8; // Solver tolerance for Schur
      real_type  convergence_tol_   = 1e-8; // Solver tolerance for Schur
    
      MatrixHandler* matrix_handler_{nullptr}; ///< Backend-specific matrix handler.
      VectorHandler* vector_handler_{nullptr}; ///< Backend-specific vector handler.

      matrix::Csr* A_{nullptr};
      vector::Vector* x_{nullptr};   // LHS of entire system
      vector::Vector* b_{nullptr};   // RHS of entire system

      real_type A_norm_ = 0.0;
      real_type b_norm_ = 0.0;
      
      // Matrices used for conjugate gradient
      matrix::Csr* A_prec_{nullptr};

      // Vectors used for conjugate gradient
      vector::Vector* d_{nullptr};
      vector::Vector* d_inv_{nullptr};
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
      
      MultiBasisParallelConjugateGradientImpl* impl_{nullptr};

      memory::MemorySpace memspace_;
    }; // class MultiBasisParallelConjugateGradient
  } // namespace hykkt
} // namespace ReSolve
