/**
 * @file LinSolverDirectCpuILU0.hpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Contains declaration of a class for incomplete LU factorization on CPU
 *
 * 
 */
#pragma once
#include "Common.hpp"
#include "LinSolver.hpp"
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

  // Forward declaration of CPU workspace
  class LinAlgWorkspaceCpu;

  class LinSolverDirectCpuILU0 : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public: 
      LinSolverDirectCpuILU0(LinAlgWorkspaceCpu* workspace = nullptr);
      ~LinSolverDirectCpuILU0();
      
      int setup(matrix::Sparse* A,
                matrix::Sparse* L = nullptr,
                matrix::Sparse* U = nullptr,
                index_type*     P = nullptr,
                index_type*     Q = nullptr,
                vector_type* rhs  = nullptr) override;
      // if values of A change, but the nnz pattern does not, redo the analysis only (reuse buffers though)
      int reset(matrix::Sparse* A);
       
      int solve(vector_type* rhs, vector_type* x) override;
      int solve(vector_type* rhs) override; // the solution is returned IN RHS (rhs is overwritten)
    

    private:
      MemoryHandler mem_; ///< Device memory manager object
      LinAlgWorkspaceCpu* workspace_{nullptr};
  };
}// namespace
