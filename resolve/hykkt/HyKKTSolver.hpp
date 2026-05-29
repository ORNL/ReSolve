#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/hykkt/ruiz/RuizScaling.hpp>
#include <resolve/hykkt/spgemm/SpGEMM.hpp>
#include <resolve/hykkt/permutation/Permutation.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
#include <resolve/hykkt/sccg/SchurComplementConjugateGradient.hpp>

namespace ReSolve
{
  using vector_type = vector::Vector;
  
  namespace hykkt
  {
    /**
     * @brief WRITE
     */
    class HyKKTSolver
    {
    public:
      HyKKTSolver(index_type nx, index_type md, index_type mc, memory::MemorySpace memspace);
      ~HyKKTSolver();

      void readMatrixFiles(
          std::istream& H_file,
          std::istream& Dx_file,
          std::istream& Ds_file,
          std::istream& J_file,
          std::istream& Jd_file,
          std::istream& rx_file,
          std::istream& rs_file,
          std::istream& ry_file,
          std::istream& ryd_file);
          
      void setMatrixBlocks(matrix::Csr* H, matrix::Csr* Dx, matrix::Csr* Ds, matrix::Csr* J, matrix::Csr* Jd);
      void setRHSBlocks(vector::Vector* rx, vector::Vector* rs, vector::Vector* ry, vector::Vector* ryd);
      void setLHSPointers(vector::Vector* x, vector::Vector* s, vector::Vector* y, vector::Vector* yd);

      void setGamma(real_type gamma);

      void addHandlers(MatrixHandler* matrixHandler, VectorHandler* vectorHandler);

      int solve();

    private:
      void setupParameters();

      // Set up solvers
      void setupSpGEMMHtil();
      void setupRuizScaling();
      void setupSpGEMMHGamma();
      void setupPermutation();
      void setupHGammaFactorization();
      void setupConjugateGradient();

      // Intermediate steps of solving the system
      void computeSpGEMMHtil();
      void computeRuizScaling();
      void computeSpGEMMHGamma();
      void applyPermutation();
      void computeHGammaFactorization(); // Uses Cholesky
      void computeConjugateGradient();
      
      void recoverSolution();
      void setupSolutionCheck();
      int checkError();

      static constexpr int ruiz_its_ = 2;
      static constexpr double norm_tol_ = 1e-2;
      static constexpr double tol_ = 1e-12;
      
      real_type gamma_; // gamma value used in HYKKT

      bool allocated_ = false;
      bool Jd_flag_   = false;

      // Whether the solver is correctly used with matrices of
      // the same nonzero structure
      bool status_    = true;

      RuizScaling* ruiz_{nullptr};
      SpGEMM* spgemm_htil_{nullptr};
      SpGEMM* spgemm_gamma_{nullptr};
      Permutation* permutation_{nullptr};
      CholeskySolver* cholesky_{nullptr};
      SchurComplementConjugateGradient* sccg_{nullptr};

      index_type mc_{0};
      index_type md_{0};
      index_type nx_{0};
      index_type n_total_{0};
      index_type m_{0};
      index_type n_{0};
      index_type N_{0}; // Size of matrix K (square) and vector x

      // Blocks of input matrix K
      matrix::Csr* H_{nullptr}; // nx x nx
      matrix::Csr* Dx_{nullptr}; // nx x nx
      matrix::Csr* Ds_{nullptr}; // md x md
      matrix::Csr* J_{nullptr}; // mc x nx
      matrix::Csr* J_tr_{nullptr}; // mx x nc
      matrix::Csr* Jd_{nullptr}; // md x nx
      matrix::Csr* Jd_tr_{nullptr}; // mx x nd

      // Blocks of input vector (RHS) r
      vector::Vector* rx_{nullptr}; // Shape: nx
      vector::Vector* rs_{nullptr}; // Shape: md
      vector::Vector* ry_{nullptr}; // Shape: mc
      vector::Vector* ryd_{nullptr}; // Shape: md

      // Blocks of output vector (LHS) x
      vector::Vector* x_{nullptr}; // Shape: nx
      vector::Vector* s_{nullptr}; // Shape: md
      vector::Vector* y_{nullptr}; // Shape: mc
      vector::Vector* yd_{nullptr}; // Shape: md
      
      // Intermediate matrices and vectors
      vector::Vector* max_d_{nullptr}; // For Ruiz scaling
      vector::Vector* rxp_{nullptr};
      vector::Vector* hrxp_{nullptr};
      vector::Vector* schur_{nullptr};
      vector::Vector* Ds_vals_{nullptr}; // = Ds_->getValues()
      vector::Vector* ryd_scaled_{nullptr};
      vector::Vector* rx_til_{nullptr};
      vector::Vector* rx_hat_{nullptr};
      vector::Vector* z_{nullptr};
      vector::Vector* ry_copy_{nullptr};
      matrix::Csr* Htil_{nullptr};
      matrix::Csr* HGam_{nullptr};
      matrix::Csr* HGam_perm_{nullptr};
      matrix::Csr* J_perm_{nullptr};
      matrix::Csr* J_tr_perm_{nullptr};
      matrix::Csr* Jd_scaled_{nullptr};
      matrix::Csr* J_copy_{nullptr};
      matrix::Csr* J_tr_copy_{nullptr};
      
      MatrixHandler* matrixHandler_{nullptr};
      VectorHandler* vectorHandler_{nullptr};
      memory::MemorySpace memspace_;
    };
  }
}