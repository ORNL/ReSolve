#pragma once
#include "Matrix.hpp"
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

      int setup_ir(resolveReal ir_tol, resolveInt ir_maxit, resolveInt ir_gs_);
      
      int solve(resolveMatrix* A, resolveVector* vec_rhs, resolveVector* vec_x);
    
    private:
      std::string first_solver_name_;
      std::string refact_solver_name_;
      std::string use_ir_;
      //IR parameters
      resolveReal ir_tol_;
      resolveInt ir_maxit_;
      resolveInt ir_gs_;

      resolveLinSolverDirect* first_solver_;
      resolveLinSolverDirect* refact_solver_;
      resolveLinSolverIterative* ir_solver_;
      bool factorization_exists_;        
  };
}
