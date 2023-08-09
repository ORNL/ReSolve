#pragma once
#include <resolve/matrix/Sparse.hpp>
#include "Vector.hpp"

namespace ReSolve
{
  RefactorizationSolver
  {
    public:
      RefactorizationSolver();
      ~RefactorizationSolver();
      int setup(std::string first_solver, 
                std::string refact_solver_, 
                std::string use_ir_);

      int setup_ir(real_type ir_tol, index_type ir_maxit, index_type ir_gs_);
      
      int solve(matrix::Sparse* A, Vector* vec_rhs, Vector* vec_x);
    
    private:
      std::string first_solver_name_;
      std::string refact_solver_name_;
      std::string use_ir_;
      //IR parameters
      real_type ir_tol_;
      index_type ir_maxit_;
      index_type ir_gs_;

      LinSolverDirect* first_solver_;
      LinSolverDirect* refact_solver_;
      LinSolverIterative* ir_solver_;
      bool factorization_exists_;        
  };
}
