#pragma once
#include "Matrix.hpp"
#include "cusolver_defs.hpp"

namespace ReSolve
{
  class MatrixCSR : public Matrix
  {
    public:
      MatrixCSR();

      MatrixCSR(Int n, Int m, Int nnz);

      MatrixCSR(Int n, 
                Int m, 
                Int nnz,
                bool symmetric,
                bool expanded);

      ~MatrixCSR();

      virtual Int* getRowData(std::string memspace);
      virtual Int* getColData(std::string memspace);
      virtual Real* getValues(std::string memspace); 

      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, std::string memspaceIn, std::string memspaceOut); 
      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, Int new_nnz, std::string memspaceIn, std::string memspaceOut); 

      virtual Int allocateMatrixData(std::string memspace); 

      virtual void print() {return;}

    private:
      Int copyCsr(std::string memspaceOut);
  };
}
