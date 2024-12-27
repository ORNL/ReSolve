#pragma once
#include <resolve/matrix/Sparse.hpp>

namespace ReSolve { namespace matrix {

  // Forward declaration of Coo
  class Coo;

  class Csr : public Sparse
  {
    public:
      Csr();

      Csr(index_type n, index_type m, index_type nnz);

      Csr(index_type n, 
          index_type m, 
          index_type nnz,
          bool symmetric,
          bool expanded);
      
      Csr(index_type n, 
          index_type m, 
          index_type nnz,
          bool symmetric,
          bool expanded,
          index_type** rows,
          index_type** cols,
          real_type** vals,
          memory::MemorySpace memspaceSrc,
          memory::MemorySpace memspaceDst);
      
      ~Csr();

      virtual index_type* getRowData(memory::MemorySpace memspace);
      virtual index_type* getColData(memory::MemorySpace memspace);
      virtual real_type*  getValues( memory::MemorySpace memspace); 

      virtual int copyData(const index_type* row_data,
                           const index_type* col_data,
                           const real_type* val_data,
                           memory::MemorySpace memspaceIn,
                           memory::MemorySpace memspaceOut);
      virtual int copyData(const index_type* row_data,
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
