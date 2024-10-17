#pragma once
#include "Common.hpp"
#include "LinSolver.hpp"
#include "cusolver_defs.hpp"
#include <resolve/MemoryUtils.hpp>
namespace ReSolve 
{
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  // Forward declaration of matrix::Sparse class
  namespace matrix
  {
    class Sparse;
  }

  // Forward declaration of ReSolve handlers workspace
  class LinAlgWorkspaceCUDA;

  class LinSolverDirectCuSolverGLU : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public:
      LinSolverDirectCuSolverGLU(LinAlgWorkspaceCUDA* workspace);
      ~LinSolverDirectCuSolverGLU();

      int refactorize() override;
      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* x) override;

      int setup(matrix::Sparse* A,
                matrix::Sparse* L,
                matrix::Sparse* U,
                index_type*     P,
                index_type*     Q,
                vector_type* rhs = nullptr) override;
    
    private:
      /// Creates M = L + U from sepeate L, U factors
      void addFactors(matrix::Sparse* L, matrix::Sparse* U);

      matrix::Sparse* M_{nullptr}; ///< the matrix that contains added factors

      // NOTE: we need cuSolver handle, we can copy it from the workspace to avoid double allocation
      cusparseMatDescr_t descr_M_{nullptr}; //this is NOT sparse matrix descriptor
      cusparseMatDescr_t descr_A_{nullptr}; //this is NOT sparse matrix descriptor
      cusolverSpHandle_t handle_cusolversp_{nullptr};

      void* glu_buffer_{nullptr};

      /// Workspace access so we can copy cusparse handle
      LinAlgWorkspaceCUDA* workspace_{nullptr};

      // Status flags
      cusolverStatus_t status_cusolver_;
      csrgluInfo_t info_M_{nullptr};

      double r_nrminf_{0.0};
      int ite_refine_succ_; 

      MemoryHandler mem_; ///< Device memory manager object
  };
}
