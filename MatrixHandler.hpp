// this class encapsulates various matrix manipulation operations, commonly required by linear solvers:
// this includes 
// (1) Matrix format conversion: coo2csr, csr2csc
// (2) Matrix vector product (SpMV)
// (3) Matrix 1-norm
#pragma once
#include "Matrix.hpp"
#include "Vector.hpp"
#include "LinAlgWorkspace.hpp"
#include <algorithm>

namespace ReSolve
{
  //helper class
  class indexPlusValue
  {
    public:
      indexPlusValue();
      ~indexPlusValue();
      void setIdx (Int new_idx);
      void setValue (Real new_value);

      Int getIdx();
      Real getValue();

      bool operator < (const indexPlusValue& str) const
      {
        return (idx_ < str.idx_);
      }  

    private:
      Int idx_;
      Real value_;
  };

  class MatrixHandler
  {
    public:
      MatrixHandler();
      MatrixHandler(LinAlgWorkspace* workspace);
      ~MatrixHandler();

      void csc2csr(Matrix* A, std::string memspace);//memspace decides on what is returned (cpu or cuda pointer)
      void coo2csr(Matrix* A, std::string memspace);

      void matvec(Matrix* A, Vector* vec_x, Vector* vec_result, Real* alpha, Real* beta, std::string memspace);
      void Matrix1Norm(Matrix *A, Real* norm);

    private: 
      LinAlgWorkspace* workspace_;
      bool new_matrix_; //if the structure changed, you need a new handler.
  };
}

