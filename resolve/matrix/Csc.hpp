#pragma once
#include "Sparse.hpp"

namespace ReSolve { namespace matrix {

  class Csc : public Sparse
  {
    public:
      Csc();
      Csc(index_type n, index_type m, index_type nnz);
      Csc(index_type n, 
          index_type m, 
          index_type nnz,
          bool symmetric,
          bool expanded);
      ~Csc();

      virtual index_type* getRowData(memory::MemorySpace memspace);
      virtual index_type* getColData(memory::MemorySpace memspace);
      virtual real_type*  getValues( memory::MemorySpace memspace); 

      virtual int updateData(const index_type* row_data,
                             const index_type* col_data,
                             const real_type* val_data,
                             memory::MemorySpace memspaceIn,
                             memory::MemorySpace memspaceOut); 
      virtual int updateData(const index_type* row_data,
                             const index_type* col_data,
                             const real_type* val_data,
                             index_type new_nnz,
                             memory::MemorySpace memspaceIn,
                             memory::MemorySpace memspaceOut); 

      virtual int allocateMatrixData(memory::MemorySpace memspace);

      virtual void print(std::ostream& file_out = std::cout, index_type indexing_base = 0);

      virtual int syncData(memory::MemorySpace memspaceOut);
  };

}} // namespace ReSolve::matrix
