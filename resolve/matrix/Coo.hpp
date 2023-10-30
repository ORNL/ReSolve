#pragma once
#include "Sparse.hpp"

namespace ReSolve { namespace matrix {

  class Coo : public Sparse
  {
    public:
      Coo();
      Coo(index_type n, index_type m, index_type nnz);
      Coo(index_type n, 
                index_type m, 
                index_type nnz,
                bool symmetric,
                bool expanded);
      ~Coo();

      virtual index_type* getRowData(memory::MemorySpace memspace);
      virtual index_type* getColData(memory::MemorySpace memspace);
      virtual real_type*  getValues( memory::MemorySpace memspace); 

      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, std::string memspaceIn, std::string memspaceOut); 
      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, std::string memspaceIn, std::string memspaceOut); 

      virtual index_type allocateMatrixData(std::string memspace);

      virtual void print();

      virtual int copyData(memory::MemorySpace memspaceOut);
  };

}} // namespace ReSolve::matrix
