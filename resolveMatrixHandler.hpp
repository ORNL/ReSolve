// this class encapsulates various matrix manipulation operations, commonly required by linear solvers:
// this includes 
// (1) Matrix format conversion: coo2csr, csr2csc
// (2) Matrix vector product (SpMV)
// (3) Matrix 1-norm
#pragma once
#include "resolveMatrix.hpp"
#include "resolveVector.hpp"
#include "resolveLinAlgWorkspace.hpp"
#include <algorithm>

namespace ReSolve
{
  //helper class
  class indexPlusValue
  {
    public:
      indexPlusValue();
      ~indexPlusValue();
      void setIdx (resolveInt new_idx);
      void setValue (resolveReal new_value);

      resolveInt getIdx();
      resolveReal getValue();

      bool operator < (const indexPlusValue& str) const
      {
        return (idx_ < str.idx_);
      }  

    private:
      resolveInt idx_;
      resolveReal value_;
  };

  class resolveMatrixHandler
  {
    public:
      resolveMatrixHandler();
      resolveMatrixHandler(resolveLinAlgWorkspace* workspace);
      ~resolveMatrixHandler();

      void csc2csr(resolveMatrix* A, std::string memspace);//memspace decides on what is returned (cpu or cuda pointer)
      void coo2csr(resolveMatrix* A, std::string memspace);

      void matvec(resolveMatrix* A, resolveVector* vec_x, resolveVector* vec_result, resolveReal* alpha, resolveReal* beta, std::string memspace);
      void resolveMatrix1Norm(resolveMatrix *A, resolveReal* norm);

    private: 
      resolveLinAlgWorkspace* workspace_;
      bool new_matrix_; //if the structure changed, you need a new handler.
  };
}

