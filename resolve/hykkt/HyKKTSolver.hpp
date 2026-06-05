/**
 * @file HyKKTSolver.hpp
 * @author Andrew Xu (xua1@ornl.gov)
 */

#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
#include <resolve/hykkt/permutation/Permutation.hpp>
#include <resolve/hykkt/ruiz/RuizScaling.hpp>
#include <resolve/hykkt/sccg/SchurComplementConjugateGradient.hpp>
#include <resolve/hykkt/spgemm/SpGEMM.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

namespace ReSolve
{
  using vector_type = vector::Vector;

  namespace hykkt
  {
    /**
     * @brief This class solves a HyKKT system defined by user-supplied matrix and RHS blocks.
     */
    class HyKKTSolver
    {
    public:
      HyKKTSolver(index_type n_x, index_type m_d, index_type m_c, memory::MemorySpace memspace);
      ~HyKKTSolver();

      void readMatrixFiles(
          std::istream& H_plus_D_x_file,
          std::istream& D_s_file,
          std::istream& J_file,
          std::istream& J_d_file,
          std::istream& r_x_file,
          std::istream& r_s_file,
          std::istream& r_y_file,
          std::istream& r_yd_file);

      void setMatrixBlocks(matrix::Csr* H_plus_D_x, matrix::Csr* D_s, matrix::Csr* J, matrix::Csr* J_d);
      void setRHSBlocks(vector::Vector* r_x, vector::Vector* r_s, vector::Vector* r_y, vector::Vector* r_yd);
      void setLHSPointers(vector::Vector* x, vector::Vector* s, vector::Vector* y, vector::Vector* y_d);

      void setGamma(real_type gamma);

      void addHandlers(MatrixHandler* matrixHandler, VectorHandler* vectorHandler);

      real_type solve();

    private:
      // Intermediate steps of solving the system
      void      setupParameters();
      void      setupSpGEMMHtilde();
      void      computeSpGEMMHtilde();
      void      setupSolutionCheck();
      void      setupRuizScaling();
      void      computeRuizScaling();
      void      setupSpGEMMHgamma();
      void      computeSpGEMMHgamma();
      void      setupPermutation();
      void      applyPermutation();
      void      setupHgammaFactorization(); // Uses Cholesky
      void      computeHgammaFactorization();
      void      setupConjugateGradient();
      void      computeConjugateGradient();
      void      recoverSolution();
      real_type checkError();

      static constexpr int    ruiz_its_     = 2;
      static constexpr double cholesky_tol_ = 1e-12;

      real_type gamma_; // gamma value used in HYKKT

      bool allocated_ = false;
      bool J_d_flag_  = false;

      // Whether the solver is correctly used with matrices of
      // the same nonzero structure
      bool status_ = true;

      RuizScaling*                      ruiz_{nullptr};
      SpGEMM*                           spgemm_htil_{nullptr};
      SpGEMM*                           spgemm_hgamma_{nullptr};
      Permutation*                      permutation_{nullptr};
      CholeskySolver*                   cholesky_{nullptr};
      SchurComplementConjugateGradient* sccg_{nullptr};

      index_type n_x_{0};
      index_type m_d_{0};
      index_type m_c_{0};
      index_type n_total_{0};

      // Blocks of input matrix K
      matrix::Csr* H_{nullptr};      // nx x nx. Actually stores H + D_x, but it is named H_ for brevity
      matrix::Csr* D_s_{nullptr};    // md x md
      matrix::Csr* J_{nullptr};      // mc x nx
      matrix::Csr* J_tr_{nullptr};   // nx x mc
      matrix::Csr* J_d_{nullptr};    // md x nx
      matrix::Csr* J_d_tr_{nullptr}; // nx x md

      // Blocks of input vector (RHS) r
      vector::Vector* r_x_{nullptr};  // Shape: nx
      vector::Vector* r_s_{nullptr};  // Shape: md
      vector::Vector* r_y_{nullptr};  // Shape: mc
      vector::Vector* r_yd_{nullptr}; // Shape: md

      // Blocks of output vector (LHS) x
      vector::Vector* x_{nullptr};   // Shape: nx
      vector::Vector* s_{nullptr};   // Shape: md
      vector::Vector* y_{nullptr};   // Shape: mc
      vector::Vector* y_d_{nullptr}; // Shape: md

      // Intermediate matrices and vectors
      vector::Vector* max_d_{nullptr}; // For Ruiz scaling
      vector::Vector* r_x_perm_{nullptr};
      vector::Vector* omega_perm_{nullptr};
      vector::Vector* schur_{nullptr};
      vector::Vector* D_s_vals_{nullptr}; // = D_s_->getValues()
      vector::Vector* r_yd_scaled_{nullptr};
      vector::Vector* r_x_til_{nullptr};
      vector::Vector* r_x_hat_{nullptr};
      vector::Vector* z_{nullptr};
      vector::Vector* r_y_copy_{nullptr};
      matrix::Csr*    H_tilde_{nullptr};
      matrix::Csr*    H_gamma_{nullptr};
      matrix::Csr*    H_gamma_perm_{nullptr};
      matrix::Csr*    J_perm_{nullptr};
      matrix::Csr*    J_tr_perm_{nullptr};
      matrix::Csr*    J_d_scaled_{nullptr};
      matrix::Csr*    J_copy_{nullptr};
      matrix::Csr*    J_tr_copy_{nullptr};

      MatrixHandler*      matrixHandler_{nullptr};
      VectorHandler*      vectorHandler_{nullptr};
      memory::MemorySpace memspace_;
    };
  } // namespace hykkt
} // namespace ReSolve
