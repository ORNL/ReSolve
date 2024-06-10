#pragma once

#include "Sparse.hpp"

namespace ReSolve
{
  namespace matrix
  {

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
        virtual real_type* getValues(memory::MemorySpace memspace);

        virtual int updateData(index_type* row_data,
                               index_type* col_data,
                               real_type* val_data,
                               memory::MemorySpace memspaceIn,
                               memory::MemorySpace memspaceOut);
        virtual int updateData(index_type* row_data,
                               index_type* col_data,
                               real_type* val_data,
                               index_type new_nnz,
                               memory::MemorySpace memspaceIn,
                               memory::MemorySpace memspaceOut);

        virtual int allocateMatrixData(memory::MemorySpace memspace);

        virtual void print(std::ostream& file_out = std::cout);

        virtual int copyData(memory::MemorySpace memspaceOut);

      private:
        // NOTE: see comment in resolve/matrix/Utilities.cpp
        virtual int expand();

        // NOTE: see comment in resolve/matrix/Utilities.cpp
        virtual std::function<
            std::tuple<std::tuple<index_type, index_type, real_type>, bool>()>
            elements(memory::MemorySpace);
    };

  } // namespace matrix
} // namespace ReSolve
