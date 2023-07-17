#pragma once
#include "Matrix.hpp"
#include "cusolver_defs.hpp"

namespace ReSolve
{
  class MatrixCSC : public Matrix
  {
    public:
      MatrixCSC();
      MatrixCSC(Int n, Int m, Int nnz);
      MatrixCSC(Int n, 
                Int m, 
                Int nnz,
                bool symmetric,
                bool expanded);
      ~MatrixCSC();

      virtual Int* getRowData(std::string memspace);
      virtual Int* getColData(std::string memspace);
      virtual Real* getValues(std::string memspace); 

      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, std::string memspaceIn, std::string memspaceOut); 
      virtual Int updateData(Int* row_data, Int* col_data, Real* val_data, Int new_nnz, std::string memspaceIn, std::string memspaceOut); 

      virtual Int allocateMatrixData(std::string memspace);

      virtual void print() {return;}

    private:
      Int  copyCsc(std::string memspaceOut);

  };
}
