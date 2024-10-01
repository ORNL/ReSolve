/**
 * @file LinSolverIterativeFGMRES.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @brief Declaration of LinSolverIterativeFGMRES class
 * 
 */
#pragma once
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include "LinSolver.hpp"
#include "GramSchmidt.hpp"

namespace ReSolve 
{
  /**
   * @brief (F)GMRES solver
   * 
   * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
   * 
   * @note MatrixHandler and VectorHandler objects are inherited from
   * LinSolver base class.
   */
  class LinSolverIterativeFGMRES : public LinSolverIterative
  {
    using vector_type = vector::Vector;

    public:
      LinSolverIterativeFGMRES(MatrixHandler* matrix_handler,
                               VectorHandler* vector_handler,
                               GramSchmidt*   gs);
      LinSolverIterativeFGMRES(index_type restart,
                               real_type  tol,
                               index_type maxit,
                               index_type conv_cond,
                               MatrixHandler* matrix_handler,
                               VectorHandler* vector_handler,
                               GramSchmidt*   gs);
      ~LinSolverIterativeFGMRES();

      int solve(vector_type* rhs, vector_type* x) override;
      int setup(matrix::Sparse* A) override;
      int resetMatrix(matrix::Sparse* new_A) override; 
      int setupPreconditioner(std::string name, LinSolverDirect* LU_solver) override;
      int setOrthogonalization(GramSchmidt* gs) override;

      int setRestart(index_type restart) override;
      int setFlexible(bool is_flexible) override;

    private:
      int allocateSolverData();
      int freeSolverData();
      void setMemorySpace();
      void precV(vector_type* rhs, vector_type* x); ///< Apply preconditioner

      memory::MemorySpace memspace_;

      vector_type* vec_V_{nullptr};
      vector_type* vec_Z_{nullptr};

      real_type* h_H_{nullptr};
      real_type* h_c_{nullptr};
      real_type* h_s_{nullptr};
      real_type* h_rs_{nullptr};

      GramSchmidt* GS_{nullptr};     
      LinSolverDirect* LU_solver_{nullptr};
      index_type n_{0};
      bool is_solver_set_{false};

      MemoryHandler mem_; ///< Device memory manager object
  };
}
