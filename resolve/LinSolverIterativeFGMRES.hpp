#pragma once
#include "Common.hpp"
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include "LinSolver.hpp"

namespace ReSolve 
{
  constexpr double ZERO = 0.0;
  constexpr double EPSILON = 1.0e-18;
  constexpr double EPSMAC  = 1.0e-16;


  class LinSolverIterativeFGMRES : public LinSolverIterative
  {
    using vector_type = vector::Vector;
    
    public:
      LinSolverIterativeFGMRES();
      LinSolverIterativeFGMRES( MatrixHandler* matrix_handler, VectorHandler* vector_handler);
      LinSolverIterativeFGMRES(index_type restart, real_type tol, index_type maxit, std::string GS_version, index_type conv_cond, MatrixHandler* matrix_handler, VectorHandler* vector_handler);
      ~LinSolverIterativeFGMRES();

      int solve(vector_type* rhs, vector_type* x);
      int setup(matrix::Sparse* A);
      int resetMatrix(matrix::Sparse* new_A); 
      int setupPreconditioner(std::string name, LinSolverDirect* LU_solver);

      real_type getTol();
      index_type getMaxit();
      index_type getRestart();
      index_type getConvCond();

      void setTol(real_type new_tol);
      void setMaxit(index_type new_maxit);
      void setRestart(index_type new_restart);
      void setGSversion(std::string new_GS);
      void setConvCond(index_type new_conv_cond);
      
      real_type getFinalResidualNorm();
      real_type getInitResidualNorm();
      index_type getNumIter();
    
    private:
      //remember matrix handler and vector handler are inherited.
      
      real_type tol_;
      index_type maxit_;
      index_type restart_;
      std::string orth_option_;
      index_type conv_cond_;
      
      real_type* d_V_;
      real_type* d_Z_;
      real_type* d_rvGPU_;
      real_type* d_Hcolumn_;
      
      real_type* h_H_;
      real_type* h_c_;
      real_type* h_s_;
      real_type* h_rs_;
      real_type* h_L_;
      real_type* h_rv_;
      real_type* h_aux_;
      real_type* d_H_col_;


      int GramSchmidt(index_type i);
      void precV(vector_type* rhs, vector_type* x); //multiply the vector by preconditioner
      LinSolverDirect* LU_solver_;
      index_type n_;// for simplicity
      real_type one_ = 1.0;
      real_type minusone_ = -1.0;
      real_type zero_ = 0.0;
      real_type final_residual_norm_;
      real_type initial_residual_norm_;
      index_type fgmres_iters_;
  };

}
