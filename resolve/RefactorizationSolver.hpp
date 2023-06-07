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

      int setup_ir(Real ir_tol, Int ir_maxit, Int ir_gs_);
      
      int solve(Matrix* A, Vector* vec_rhs, Vector* vec_x);
    
    private:
      std::string first_solver_name_;
      std::string refact_solver_name_;
      std::string use_ir_;
      //IR parameters
      Real ir_tol_;
      Int ir_maxit_;
      Int ir_gs_;

      LinSolverDirect* first_solver_;
      LinSolverDirect* refact_solver_;
      LinSolverIterative* ir_solver_;
      bool factorization_exists_;        
  };
}
