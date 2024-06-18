#include <algorithm>
#include <cassert>
#include <memory>

#include "Utilities.hpp"

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  int matrix::expand(matrix::Coo& A)
  {
    if (A.symmetric() && !A.expanded()) {
      index_type* rows = A.getRowData(memory::HOST);
      index_type* columns = A.getColData(memory::HOST);
      real_type* values = A.getValues(memory::HOST);

      if (rows == nullptr || columns == nullptr || values == nullptr) {
        return 0;
      }

      // NOTE: this is predicated on the same define as that which disables
      //       assert(3), to avoid record-keeping where it is not necessary
#ifndef NDEBUG
      index_type n_diagonal = 0;
#endif

      index_type nnz_expanded = A.getNnzExpanded();

      // NOTE: so because most of the code here uses new/delete and there's no
      //       realloc(3) equivalent for that memory management scheme, we
      //       have to manually new/memcpy/delete, unfortunately
      index_type* new_rows = new index_type[nnz_expanded];
      index_type* new_columns = new index_type[nnz_expanded];
      real_type* new_values = new real_type[nnz_expanded];

      index_type nnz = A.getNnz();
      index_type j = 0;

      for (index_type i = 0; i < nnz; i++) {
        new_rows[j] = rows[i];
        new_columns[j] = columns[i];
        new_values[j] = values[i];

        j++;

#ifndef NDEBUG
        if (rows[i] == columns[i]) {
          n_diagonal++;
        } else {
#else
        if (rows[i] != columns[i]) {
#endif
          new_rows[j] = columns[i];
          new_columns[j] = rows[i];
          new_values[j] = values[i];

          j++;
        }
      }

      // NOTE: the effectiveness of this is probably questionable given that
      //       it occurs after we've already risked writing out-of-bounds, but
      //       i guess if that worked or we've over-allocated, this will catch
      //       something (in debug builds/release builds with asserts enabled)
      assert(nnz_expanded == ((2 * nnz) - n_diagonal));

      if (A.destroyMatrixData(memory::HOST) != 0 ||
          A.setMatrixData(new_rows, new_columns, new_values, memory::HOST) != 0) {
        // TODO: make fallible
        assert(false && "invalid state after coo matrix expansion");
        return -1;
      }

      A.setNnz(nnz_expanded);
      A.setExpanded(true);
      A.setDataOwnership(true, memory::HOST);
      A.setValueOwnership(true, memory::HOST);
    }

    return 0;
  }

  int matrix::expand(matrix::Csr& A)
  {
    if (A.symmetric() && !A.expanded()) {
      index_type* rows = A.getRowData(memory::HOST);
      index_type* columns = A.getColData(memory::HOST);
      real_type* values = A.getValues(memory::HOST);

      if (rows == nullptr || columns == nullptr || values == nullptr) {
        return 0;
      }

      index_type n_rows = A.getNumRows();
      index_type* new_rows = new index_type[n_rows + 1];
      index_type new_i;
      std::unique_ptr<index_type[]> remaining(new index_type[n_rows]);
      std::fill_n(remaining.get(), n_rows, 0);

      // allocation

      for (index_type row = 0; row < n_rows; row++) {
        for (index_type i = rows[row]; i < rows[row + 1]; i++) {
          if (columns[i] != row) {
            remaining[columns[i]]++;
          }
        }
      }

      new_rows[0] = 0;
      for (index_type row = 0; row < n_rows; row++) {
        new_rows[row + 1] =
            new_rows[row] + remaining[row] + rows[row + 1] - rows[row];
      }

      assert(A.getNnzExpanded() == new_rows[n_rows]);

#ifdef NDEBUG
      // TODO: should this be here? is this a good idea?
      A.setNnzExpanded(new_rows[n_rows]);
#endif

      index_type nnz_expanded = A.getNnzExpanded();
      index_type* new_columns = new index_type[nnz_expanded];
      real_type* new_values = new real_type[nnz_expanded];

      // fill

      for (index_type row = 0; row < n_rows; row++) {
        new_i = new_rows[row];
        for (index_type i = rows[row]; i < rows[row + 1]; i++) {
          new_columns[new_i] = columns[i];
          new_values[new_i] = values[i];

          if (columns[i] != row) {
            index_type o = new_rows[columns[i] + 1] - remaining[columns[i]]--;
            new_columns[o] = row;
            new_values[o] = values[i];
          }

          new_i++;
        }
      }

      if (A.destroyMatrixData(memory::HOST) != 0 ||
          A.setMatrixData(new_rows, new_columns, new_values, memory::HOST) != 0) {
        assert(false && "invalid state after csr matrix expansion");
        return -1;
      }

      A.setNnz(nnz_expanded);
      A.setExpanded(true);
      A.setDataOwnership(true, memory::HOST);
      A.setValueOwnership(true, memory::HOST);
    }

    return 0;
  }
} // namespace ReSolve
