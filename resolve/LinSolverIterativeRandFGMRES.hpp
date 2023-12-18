#pragma once
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include "LinSolver.hpp"
#include "GramSchmidt.hpp"
#include "RandSketchingManager.hpp"

namespace ReSolve
{
  /**
   * @brief Randomized (F)GMRES
   * 
   * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
   * 
   * @note MatrixHandler and VectorHandler objects are inherited from
   * LinSolver base class.
   * 
   */
  class LinSolverIterativeRandFGMRES : public LinSolverIterative
  {
    private:
      using vector_type = vector::Vector;

    public:
      enum SketchingMethod { cs = 0,    // count sketch 
                             fwht = 1}; // fast Walsh-Hadamard transform
    
      LinSolverIterativeRandFGMRES(std::string memspace = "cuda");

      LinSolverIterativeRandFGMRES(MatrixHandler* matrix_handler,
                                   VectorHandler* vector_handler,
                                   SketchingMethod rand_method, 
                                   GramSchmidt*   gs,
                                   std::string memspace = "cuda");

      LinSolverIterativeRandFGMRES(index_type restart,
                                   real_type  tol,
                                   index_type maxit,
                                   index_type conv_cond,
                                   MatrixHandler* matrix_handler,
                                   VectorHandler* vector_handler, 
                                   SketchingMethod rand_method, 
                                   GramSchmidt*   gs,
                                   std::string memspace = "cuda");

      ~LinSolverIterativeRandFGMRES();

      int solve(vector_type* rhs, vector_type* x) override;
      int setup(matrix::Sparse* A) override;
      int resetMatrix(matrix::Sparse* new_A) override; 
      int setupPreconditioner(std::string name, LinSolverDirect* LU_solver) override;

      index_type getKrand();

    private:
      memory::MemorySpace memspace_;

      vector_type* d_V_{nullptr};
      vector_type* d_Z_{nullptr};
      // for performing Gram-Schmidt
      vector_type* d_S_{nullptr};

      real_type* h_H_{nullptr};
      real_type* h_c_{nullptr};
      real_type* h_s_{nullptr};
      real_type* h_rs_{nullptr};
      real_type* d_aux_{nullptr};

      GramSchmidt* GS_;     
      void precV(vector_type* rhs, vector_type* x); ///< multiply the vector by preconditioner
      LinSolverDirect* LU_solver_;
      index_type n_;
      real_type one_over_k_{1.0};

      index_type k_rand_{0}; ///< size of sketch space. We need to know it so we can allocate S!
      MemoryHandler mem_;    ///< Device memory manager object
      RandSketchingManager* rand_manager_{nullptr};
      SketchingMethod rand_method_; 
  };
} // namespace ReSolve
