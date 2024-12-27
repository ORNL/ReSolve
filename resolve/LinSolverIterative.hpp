/**
 * @file LinSolverIterative.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of iterative solver base class.
 * 
 */
#pragma once

#include <string>
#include <resolve/LinSolver.hpp>

namespace ReSolve 
{
  class GramSchmidt;

  class LinSolverIterative : public LinSolver 
  {
    public:
      LinSolverIterative();
      virtual ~LinSolverIterative();
      virtual int setup(matrix::Sparse* A);
      virtual int resetMatrix(matrix::Sparse* A) = 0;
      virtual int setupPreconditioner(std::string type, LinSolverDirect* LU_solver) = 0;

      virtual int  solve(vector_type* rhs, vector_type* init_guess) = 0;

      virtual real_type getFinalResidualNorm() const;
      virtual real_type getInitResidualNorm() const;
      virtual index_type getNumIter() const;

      virtual int setOrthogonalization(GramSchmidt* gs);

      real_type getTol() const;
      index_type getMaxit() const;

      void setTol(real_type new_tol);
      void setMaxit(index_type new_maxit);

    protected:
      real_type initial_residual_norm_;
      real_type final_residual_norm_;
      index_type total_iters_;

      // Parameters common for all iterative solvers
      real_type tol_{1e-14};  ///< Solver tolerance
      index_type maxit_{100}; ///< Maximum solver iterations
  };
}
