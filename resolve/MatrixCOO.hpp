#pragma once
#include "Matrix.hpp"

namespace ReSolve
{
  class MatrixCOO : public Matrix
  {
    public:
      MatrixCOO();
      MatrixCOO(index_type n, index_type m, index_type nnz);
      MatrixCOO(index_type n, 
                index_type m, 
                index_type nnz,
                bool symmetric,
                bool expanded);
      ~MatrixCOO();

      virtual index_type* getRowData(std::string memspace);
      virtual index_type* getColData(std::string memspace);
      virtual real_type* getValues(std::string memspace); 

      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspaceIn, std::string memspaceOut); 
      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, std::string memspaceIn, std::string memspaceOut); 

      virtual index_type allocateMatrixData(std::string memspace);

      virtual void print();

    private:

      index_type copyCoo(std::string memspaceOut);
  };
}
