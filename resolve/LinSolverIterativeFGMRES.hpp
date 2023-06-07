#pragma once
#include "Common.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "LinSolver.hpp"

namespace ReSolve 
{
  constexpr double ZERO = 0.0;
  constexpr double EPSILON = 1.0e-18;
  constexpr double EPSMAC  = 1.0e-16;


  class LinSolverIterativeFGMRES : public LinSolverIterative
  {
    public:
      LinSolverIterativeFGMRES();
      LinSolverIterativeFGMRES( MatrixHandler* matrix_handler, VectorHandler* vector_handler);
      LinSolverIterativeFGMRES(Int restart, Real tol, Int maxit, std::string GS_version, Int conv_cond, MatrixHandler* matrix_handler, VectorHandler* vector_handler);
      ~LinSolverIterativeFGMRES();

      int solve(Vector* rhs, Vector* x);
      void setup(Matrix* A);
      int resetMatrix(Matrix* new_A); 
      int setupPreconditioner(std::string name, LinSolverDirect* LU_solver);

      Real getTol();
      Int getMaxit();
      Int getRestart();
      Int getConvCond();

      void setTol(Real new_tol);
      void setMaxit(Int new_maxit);
      void setRestart(Int new_restart);
      void setGSversion(std::string new_GS);
      void setConvCond(Int new_conv_cond);
      
      Real getFinalResidualNorm();
      Real getInitResidualNorm();
      Int getNumIter();
    
    private:
      //remember matrix handler and vector handler are inherited.
      
      Real tol_;
      Int maxit_;
      Int restart_;
      std::string orth_option_;
      Int conv_cond_;
      
      Real* d_V_;
      Real* d_Z_;
      Real* d_rvGPU_;
      Real* d_Hcolumn_;
      
      Real* h_H_;
      Real* h_c_;
      Real* h_s_;
      Real* h_rs_;
      Real* h_L_;
      Real* h_rv_;
      Real* h_aux_;
      Real* d_H_col_;


      int GramSchmidt(Int i);
      void precV(Vector* rhs, Vector* x); //multiply the vector by preconditioner
      LinSolverDirect* LU_solver_;
      Int n_;// for simplicity
      Real one_ = 1.0;
      Real minusone_ = -1.0;
      Real zero_ = 0.0;
      Real final_residual_norm_;
      Real initial_residual_norm_;
      Int fgmres_iters_;
  };

}
