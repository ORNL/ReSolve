#include <algorithm>
#include <cassert>
#include <memory>

#include "Utilities.hpp"

#include <resolve/Common.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/utilities/misc/IndexValuePair.hpp>

namespace ReSolve
{
  using out = io::Logger;
  namespace matrix
  {
    /**
     * @brief Creates a CSR from a COO matrix.
     *
     * @param[in]  A
     * @param[out] B
     * @return int - Error code, 0 if successful.
     *
     * @pre `A` is a valid sparse matrix in unordered COO format. Duplicates are allowed.
     * Up-to-date values and indices must be on the host.
     *
     * @post `B` represents the same matrix as `A` but is in the CSR format. `B` is
     * allocated and stored on the host.
     *
     * @invariant `A` is not changed.
     */
    int coo2csr(matrix::Coo* A, matrix::Csr* B, memory::MemorySpace memspace)
    {
      index_type* rows = A->getRowData(memory::HOST);
      index_type* columns = A->getColData(memory::HOST);
      real_type* values = A->getValues(memory::HOST);

      if (rows == nullptr || columns == nullptr || values == nullptr) {
        return 0;
      }

      index_type nnz = A->getNnz();
      index_type n_rows = A->getNumRows();
      index_type* new_rows = new index_type[n_rows + 1];
      std::fill_n(new_rows, n_rows + 1, 0);

      // NOTE: this is the only auxiliary storage buffer used by this conversion
      //       function. it is first used to track the number of values on the
      //       diagonal (if the matrix is symmetric and unexpanded), then it is
      //       used to track the amount of spaces used in each row's value and
      //       column data
      std::unique_ptr<index_type[]> used(new index_type[n_rows]);
      std::fill_n(used.get(), n_rows, 0);

      // allocation stage, the first loop is O(nnz) no matter the branch and the second
      // is O(n)
      //
      // all this does is prepare the row index array based on the nonzeroes stored in
      // the input coo matrix. if the matrix is symmetric and either upper or lower
      // triangular, it additionally corrects for mirrored values using the `used` array
      // and validates the triangularity. the branch is done to avoid the extra work if
      // it's not necessary

      if (!A->symmetric() || A->expanded()) {
        for (index_type i = 0; i < nnz; i++) {
          new_rows[rows[i] + 1]++;
        }
      } else {
        bool upper_triangular = false;
        for (index_type i = 0; i < nnz; i++) {
          new_rows[rows[i] + 1]++;
          if (rows[i] != columns[i]) {
            used[columns[i]]++;

            if (rows[i] > columns[i] && upper_triangular) {
              assert(false && "a matrix indicated to be symmetric triangular was not actually symmetric triangular");
              return -1;
            }
            upper_triangular = rows[i] < columns[i];
          }
        }
      }

      for (index_type row = 0; row < n_rows; row++) {
        new_rows[row + 1] += new_rows[row] + used[row];
        used[row] = 0;
      }

      index_type* new_columns = new index_type[new_rows[n_rows]];
      std::fill_n(new_columns, new_rows[n_rows], -1);
      real_type* new_values = new real_type[new_rows[n_rows]];

      // fill stage, approximately O(nnz * m) in the worst case
      //
      // all this does is iterate over the nonzeroes in the coo matrix,
      // check to see if a value at that colum already exists using binary search,
      // and if it does, then insert the new value at that position (deduplicating
      // the matrix), otherwise, it allocates a new spot in the row (where you see
      // used[rows[i]]++) and shifts everything over, performing what is effectively
      // insertion sort. the lower half is conditioned on the matrix being symmetric
      // and stored as either upper-triangular or lower-triangular, and just
      // performs the same as what is described above, but with the indices swapped.

      for (index_type i = 0; i < nnz; i++) {
        index_type insertion_pos =
            static_cast<index_type>(
                std::lower_bound(&new_columns[new_rows[rows[i]]],
                                 &new_columns[new_rows[rows[i]] + used[rows[i]]],
                                 columns[i])
                - new_columns);

        if (new_columns[insertion_pos] == columns[i]) {
          new_values[insertion_pos] = values[i];
        } else {
          for (index_type offset = new_rows[rows[i]] + used[rows[i]]++;
               offset > insertion_pos;
               offset--) {
            std::swap(new_columns[offset], new_columns[offset - 1]);
            std::swap(new_values[offset], new_values[offset - 1]);
          }

          new_columns[insertion_pos] = columns[i];
          new_values[insertion_pos] = values[i];
        }

        if ((A->symmetric() && !A->expanded()) && (columns[i] != rows[i])) {
          index_type mirrored_insertion_pos =
              static_cast<index_type>(
                  std::lower_bound(&new_columns[new_rows[columns[i]]],
                                   &new_columns[new_rows[columns[i]] + used[columns[i]]],
                                   rows[i])
                  - new_columns);

          if (new_columns[mirrored_insertion_pos] == rows[i]) {
            new_values[mirrored_insertion_pos] = values[i];
          } else {
            for (index_type offset = new_rows[columns[i]] + used[columns[i]]++;
                 offset > mirrored_insertion_pos;
                 offset--) {
              std::swap(new_columns[offset], new_columns[offset - 1]);
              std::swap(new_values[offset], new_values[offset - 1]);
            }

            new_columns[mirrored_insertion_pos] = rows[i];
            new_values[mirrored_insertion_pos] = values[i];
          }
        }
      }

      // backshifting stage, approximately O(nnz * m) in the worst case
      //
      // all this does is shift back rows to remove empty space in between them
      // by indexing each row in order, checking to see if the amount of used
      // spaces is equivalent to the amount of nonzeroes in the row, and if not,
      // shifts every subsequent row back the amount of unused spaces

      for (index_type row = 0; row < n_rows - 1; row++) {
        index_type row_nnz = new_rows[row + 1] - new_rows[row];
        if (used[row] != row_nnz) {
          index_type correction = row_nnz - used[row];

          for (index_type corrected_row = row + 1;
               corrected_row < n_rows;
               corrected_row++) {
            for (index_type offset = new_rows[corrected_row];
                 offset < new_rows[corrected_row + 1];
                 offset++) {
              new_columns[offset - correction] = new_columns[offset];
              new_values[offset - correction] = new_values[offset];
            }

            new_rows[corrected_row] -= correction;
          }

          new_rows[n_rows] -= correction;
        }
      }

      B->setSymmetric(A->symmetric());
      B->setNnz(new_rows[n_rows]);
      // NOTE: this is necessary because updateData always reads the current nnz from
      //       this field. see #176
      B->setNnzExpanded(new_rows[n_rows]);
      B->setExpanded(true);

      if (B->updateData(new_rows, new_columns, new_values, memory::HOST, memspace) != 0) {
        delete[] new_rows;
        delete[] new_columns;
        delete[] new_values;

        assert(false && "invalid state after coo -> csr conversion");
        return -1;
      }

      delete[] new_rows;
      delete[] new_columns;
      delete[] new_values;

      return 0;
    }
  } // namespace matrix
} // namespace ReSolve
