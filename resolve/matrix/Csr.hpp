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
      
      Csr(matrix::Coo* mat, memory::MemorySpace memspace);

      ~Csr();

      virtual index_type* getRowData(memory::MemorySpace memspace);
      virtual index_type* getColData(memory::MemorySpace memspace);
      virtual real_type*  getValues( memory::MemorySpace memspace); 

      virtual int updateData(index_type* row_data, index_type* col_data, real_type* val_data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut); 
      virtual int updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut);

      virtual int allocateMatrixData(memory::MemorySpace memspace); 

      virtual void print() {return;}

      virtual int copyData(memory::MemorySpace memspaceOut);

      int updateFromCoo(matrix::Coo* mat, memory::MemorySpace memspaceOut);

    private:
      int coo2csr(matrix::Coo* mat, memory::MemorySpace memspace);
  };

}} // namespace ReSolve::matrix
