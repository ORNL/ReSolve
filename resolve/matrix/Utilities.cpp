#include <algorithm>
#include <cassert>
#include <memory>

#include "Utilities.hpp"

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  /**
   * @invariant The input matrix must be deduplicated, otherwise the result is undefined
   */
  int matrix::expand(matrix::Coo& A)
  {
    if (A.symmetric() && !A.expanded()) {
      index_type* rows = A.getRowData(memory::HOST);
      index_type* columns = A.getColData(memory::HOST);
      real_type* values = A.getValues(memory::HOST);

      if (rows == nullptr || columns == nullptr || values == nullptr) {
        return 0;
      }

      index_type nnz = A.getNnz();
      index_type n_diagonal = 0;
      for (index_type i = 0; i < nnz; i++) {
        if (rows[i] == columns[i]) {
          n_diagonal++;
        }
      }

      index_type nnz_expanded = (2 * nnz) - n_diagonal;
      A.setNnzExpanded(nnz_expanded);

      // NOTE: so because most of the code here uses new/delete and there's no
      //       realloc(3) equivalent for that memory management scheme, we
      //       have to manually new/memcpy/delete, unfortunately
      index_type* new_rows = new index_type[nnz_expanded];
      index_type* new_columns = new index_type[nnz_expanded];
      real_type* new_values = new real_type[nnz_expanded];

      index_type j = 0;

      for (index_type i = 0; i < nnz; i++) {
        new_rows[j] = rows[i];
        new_columns[j] = columns[i];
        new_values[j] = values[i];

        j++;

        if (rows[i] != columns[i]) {
          new_rows[j] = columns[i];
          new_columns[j] = rows[i];
          new_values[j] = values[i];

          j++;
        }
      }

      if (A.destroyMatrixData(memory::HOST) != 0 ||
          A.setMatrixData(new_rows, new_columns, new_values, memory::HOST) != 0) {
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

  /**
   * @invariant The input matrix must be deduplicated, otherwise the result is undefined
   */
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

      A.setNnzExpanded(new_rows[n_rows]);
      index_type nnz_expanded = new_rows[n_rows];
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

  /**
   * @invariant The input matrix must be deduplicated, otherwise the result is undefined
   */
  int matrix::expand(matrix::Csc& A)
  {
    if (A.symmetric() && !A.expanded()) {
      index_type* rows = A.getRowData(memory::HOST);
      index_type* columns = A.getColData(memory::HOST);
      real_type* values = A.getValues(memory::HOST);

      if (rows == nullptr || columns == nullptr || values == nullptr) {
        return 0;
      }

      index_type n_columns = A.getNumColumns();
      index_type* new_columns = new index_type[n_columns + 1];
      index_type new_i;
      std::unique_ptr<index_type[]> remaining(new index_type[n_columns]);
      std::fill_n(remaining.get(), n_columns, 0);

      // allocation

      for (index_type column = 0; column < n_columns; column++) {
        for (index_type i = columns[column]; i < columns[column + 1]; i++) {
          if (rows[i] != column) {
            remaining[rows[i]]++;
          }
        }
      }

      new_columns[0] = 0;
      for (index_type column = 0; column < n_columns; column++) {
        new_columns[column + 1] =
            new_columns[column] + remaining[column] + columns[column + 1] - columns[column];
      }

      A.setNnzExpanded(new_columns[n_columns]);
      index_type nnz_expanded = new_columns[n_columns];
      index_type* new_rows = new index_type[nnz_expanded];
      real_type* new_values = new real_type[nnz_expanded];

      // fill

      for (index_type column = 0; column < n_columns; column++) {
        new_i = new_columns[column];
        for (index_type i = columns[column]; i < columns[column + 1]; i++) {
          new_rows[new_i] = rows[i];
          new_values[new_i] = values[i];

          if (rows[i] != column) {
            index_type o = new_columns[rows[i] + 1] - remaining[rows[i]]--;
            new_rows[o] = column;
            new_values[o] = values[i];
          }

          new_i++;
        }
      }

      if (A.destroyMatrixData(memory::HOST) != 0 ||
          A.setMatrixData(new_rows, new_columns, new_values, memory::HOST) != 0) {
        assert(false && "invalid state after csc matrix expansion");
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
