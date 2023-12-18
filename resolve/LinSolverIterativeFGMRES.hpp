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
      LinSolverIterativeFGMRES(std::string memspace = "cuda");
      LinSolverIterativeFGMRES(MatrixHandler* matrix_handler,
                              VectorHandler* vector_handler,
                              GramSchmidt*   gs,
                              std::string memspace = "cuda");
      LinSolverIterativeFGMRES(index_type restart,
                              real_type  tol,
                              index_type maxit,
                              index_type conv_cond,
                              MatrixHandler* matrix_handler,
                              VectorHandler* vector_handler,
                              GramSchmidt*   gs,
                              std::string memspace = "cuda");
      ~LinSolverIterativeFGMRES();

      int solve(vector_type* rhs, vector_type* x) override;
      int setup(matrix::Sparse* A) override;
      int resetMatrix(matrix::Sparse* new_A) override; 
      int setupPreconditioner(std::string name, LinSolverDirect* LU_solver) override;

    private:
      memory::MemorySpace memspace_;

      std::string orth_option_;
      vector_type* d_V_{nullptr};
      vector_type* d_Z_{nullptr};

      real_type* h_H_{nullptr};
      real_type* h_c_{nullptr};
      real_type* h_s_{nullptr};
      real_type* h_rs_{nullptr};


      GramSchmidt* GS_;     
      void precV(vector_type* rhs, vector_type* x); //multiply the vector by preconditioner
      LinSolverDirect* LU_solver_;
      index_type n_;// for simplicity

      MemoryHandler mem_; ///< Device memory manager object
  };
}
