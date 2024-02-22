
#pragma once
#include "Common.hpp"
#include "LinSolver.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

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

  class LinSolverDirectSerialILU0 : public LinSolverDirect 
  {
    using vector_type = vector::Vector;
    
    public: 
      LinSolverDirectSerialILU0(LinAlgWorkspaceCpu* workspace);
      ~LinSolverDirectSerialILU0();
      
      int setup(matrix::Sparse* A,
                matrix::Sparse* L = nullptr,
                matrix::Sparse* U = nullptr,
                index_type*     P = nullptr,
                index_type*     Q = nullptr,
                vector_type* rhs  = nullptr);
      int reset(matrix::Sparse* A);
       
      int solve(vector_type* rhs, vector_type* x);
      int solve(vector_type* rhs);// the solutuon is returned IN RHS (rhs is overwritten)
    

    private:

      MemoryHandler mem_; ///< Device memory manager object
      LinAlgWorkspaceCpu* workspace_{nullptr}; 
      bool owns_factors_{false};    ///< If the class owns L and U factors

      real_type* h_aux1_{nullptr};
      // since ILU OVERWRITES THE MATRIX values, we need a buffer to keep 
      // the values of ILU decomposition. 
      real_type* h_ILU_vals_{nullptr};
  };
}// namespace
