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

      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut); 
      virtual index_type updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut); 

      virtual index_type allocateMatrixData(memory::MemorySpace memspace);

      virtual void print();

      virtual int copyData(memory::MemorySpace memspaceOut);
  };

}} // namespace ReSolve::matrix
