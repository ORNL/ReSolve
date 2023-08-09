// this class encapsulates various matrix manipulation operations, commonly required by linear solvers:
// this includes 
// (1) Matrix format conversion: coo2csr, csr2csc
// (2) Matrix vector product (SpMV)
// (3) Matrix 1-norm
#pragma once
#include "Csr.hpp"
#include "Csc.hpp"
#include "Coo.hpp"
#include <resolve/Vector.hpp>
#include <resolve/LinAlgWorkspace.hpp>
#include <algorithm>

namespace ReSolve { namespace matrix {

  //helper class
  class indexPlusValue
  {
    public:
      indexPlusValue();
      ~indexPlusValue();
      void setIdx (index_type new_idx);
      void setValue (real_type new_value);

      index_type getIdx();
      real_type getValue();

      bool operator < (const indexPlusValue& str) const
      {
        return (idx_ < str.idx_);
      }  

    private:
      index_type idx_;
      real_type value_;
  };

  class MatrixHandler
  {
    public:
      MatrixHandler();
      MatrixHandler(LinAlgWorkspace* workspace);
      ~MatrixHandler();

      index_type csc2csr(Csc* A_csr, Csr* A_csc, std::string memspace);//memspace decides on what is returned (cpu or cuda pointer)
      void coo2csr(Coo* A_coo, Csr* A_csr, std::string memspace);

      int matvec(Sparse* A, Vector* vec_x, Vector* vec_result, real_type* alpha, real_type* beta,std::string matrix_type, std::string memspace);
      void Matrix1Norm(Sparse *A, real_type* norm);
     bool getValuesChanged();
     void setValuesChanged(bool toWhat); 
    
    private: 
      LinAlgWorkspace* workspace_;
      bool new_matrix_; //if the structure changed, you need a new handler.
      bool values_changed_; // needed for matvec
  };

}} // namespace ReSolve::matrix

