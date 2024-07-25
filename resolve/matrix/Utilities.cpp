#include <algorithm>
#include <cassert>
#include <memory>

#include "Utilities.hpp"

#include <resolve/Common.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>

namespace ReSolve
{
  namespace matrix
  {
    /**
     * @param[in] A
     * @param[out] B
     * @param[in] memspace
     * @return int - Error code, 0 if successful
     *
     * @pre `A` is a valid sparse matrix in an assumed to be unordered COO format.
     *      Duplicates are permitted, and the values and indices on the host must
     *      be up to date
     *
     * @post `B` is semantically the same matrix as `A`, but has been deduplicated,
     *       has been expanded if `A` was symmetric and either upper or lower
     *       triangular, and has had its triplets sorted in a "column-major" manner
     *
     * @invariant `A` is unchanged
     */
    int coo2coo(matrix::Coo* A, matrix::Coo* B, memory::MemorySpace memspace)
    {
      index_type* a_rows = A->getRowData(memory::HOST);
      index_type* a_columns = A->getColData(memory::HOST);
      real_type* a_values = A->getValues(memory::HOST);

      if (a_rows == nullptr || a_columns == nullptr || a_values == nullptr) {
        return 0;
      }

      index_type nnz_with_duplicates = A->getNnz();
      index_type n_columns = A->getNumColumns();

      // NOTE: auxiliary memory that is first used to track the amount
      //       of off-diagonal elements to perform space allocation when
      //       the matrix is symmetric and unexpanded, then is used to
      //       track the amount of space used in each column to allow
      //       out of order filling of values
      std::unique_ptr<index_type[]> used(new index_type[n_columns]);
      std::fill_n(used.get(), n_columns, 0);

      // NOTE: column partitions of the destination matrices, used
      //       to allow out of order filling of values (as is done
      //       during expansion of the input matrix)
      std::unique_ptr<index_type[]> partitions(new index_type[n_columns + 1]);
      std::fill_n(partitions.get(), n_columns + 1, 0);

      // allocation stage, the first loop is O(nnz) always and the second is O(n)
      //
      // computes the nnz of the destination matrix and allocates space for each
      // column. additionally, the upper or lower diagonality is checked if the
      // input matrix is symmetric and unexpanded

      if (!A->symmetric() || A->expanded()) {
        for (index_type i = 0; i < nnz_with_duplicates; i++) {
          partitions[a_columns[i] + 1]++;
        }
      } else {
        bool is_upper_triangular = false;
        bool is_lower_triangular = false;

        for (index_type i = 0; i < nnz_with_duplicates; i++) {
          partitions[a_columns[i] + 1]++;

          if (a_rows[i] != a_columns[i]) {
            used[a_rows[i]]++;

            is_upper_triangular |= a_rows[i] < a_columns[i];
            is_lower_triangular |= a_rows[i] > a_columns[i];
            if (is_upper_triangular && is_lower_triangular) {
              assert(false && "a matrix indicated to be symmetric triangular was not actually symmetric triangular");
              return -1;
            }
          }
        }
      }

      for (index_type column = 0; column < n_columns; column++) {
        partitions[column + 1] += partitions[column] + used[column];
        used[column] = 0;
      }

      index_type new_nnz_with_duplicates = partitions[n_columns];
      index_type* b_rows = new index_type[new_nnz_with_duplicates];
      std::fill_n(b_rows, new_nnz_with_duplicates, -1);
      index_type* b_columns = new index_type[new_nnz_with_duplicates];
      real_type* b_values = new real_type[new_nnz_with_duplicates];

      // fill stage, approximately O(nnz * n) in the worst case
      //
      // all this does is iterate over the nonzeroes in the input matrix,
      // check to see if a value at that column already exists using binary search,
      // and if it does, then add to the value at that position, deduplicating the
      // matrix. otherwise, it allocates a new spot in the row (where you see
      // used[a_rows[i]]++) and shifts everything over, performing what is
      // effectively insertion sort. the lower half is conditioned on the matrix
      // being symmetric and stored as either upper-triangular or lower-triangular,
      // and just performs the same as what is described above, but with the
      // indices swapped.

      for (index_type i = 0; i < nnz_with_duplicates; i++) {
        // this points to the first element not less than `a_rows[i]`. see
        // https://en.cppreference.com/w/cpp/algorithm/lower_bound for more details
        index_type* closest_position =
            std::lower_bound(&b_rows[partitions[a_columns[i]]],
                             &b_rows[partitions[a_columns[i]] + used[a_columns[i]]],
                             a_rows[i]);

        // this is the offset at which the element's value belongs
        index_type insertion_offset = static_cast<index_type>(closest_position - b_rows);

        if (b_rows[insertion_offset] == a_rows[i]) {
          b_values[insertion_offset] += a_values[i];
        } else {
          for (index_type offset = partitions[a_columns[i]] + used[a_columns[i]]++;
               offset > insertion_offset;
               offset--) {
            std::swap(b_rows[offset], b_rows[offset - 1]);
            std::swap(b_columns[offset], b_columns[offset - 1]);
            std::swap(b_values[offset], b_values[offset - 1]);
          }

          b_rows[insertion_offset] = a_rows[i];
          b_columns[insertion_offset] = a_columns[i];
          b_values[insertion_offset] = a_values[i];
        }

        if ((A->symmetric() && !A->expanded()) && (a_columns[i] != a_rows[i])) {
          index_type* mirrored_closest_position =
              std::lower_bound(&b_rows[partitions[a_rows[i]]],
                               &b_rows[partitions[a_rows[i]] + used[a_rows[i]]],
                               a_columns[i]);
          index_type mirrored_insertion_offset = static_cast<index_type>(mirrored_closest_position - b_rows);

          if (b_rows[mirrored_insertion_offset] == a_columns[i]) {
            b_values[mirrored_insertion_offset] += a_values[i];
          } else {
            for (index_type offset = partitions[a_rows[i]] + used[a_rows[i]]++;
                 offset > mirrored_insertion_offset;
                 offset--) {
              std::swap(b_rows[offset], b_rows[offset - 1]);
              std::swap(b_columns[offset], b_columns[offset - 1]);
              std::swap(b_values[offset], b_values[offset - 1]);
            }

            b_rows[mirrored_insertion_offset] = a_columns[i];
            b_columns[mirrored_insertion_offset] = a_rows[i];
            b_values[mirrored_insertion_offset] = a_values[i];
          }
        }
      }

      // backshifting stage, approximately O(nnz * m) in the worst case
      //
      // all this does is shift back rows to remove empty space in between them
      // by indexing each row in order, checking to see if the amount of used
      // spaces is equivalent to the amount of nonzeroes in the row, and if not,
      // shifts every subsequent row back the amount of unused spaces

      for (index_type column = 0; column < n_columns; column++) {
        index_type column_nnz = partitions[column + 1] - partitions[column];
        if (used[column] != column_nnz) {
          index_type correction = column_nnz - used[column];

          for (index_type corrected_column = column + 1;
               corrected_column < n_columns;
               corrected_column++) {
            for (index_type offset = partitions[corrected_column];
                 offset < partitions[corrected_column + 1];
                 offset++) {
              b_rows[offset - correction] = b_rows[offset];
              b_columns[offset - correction] = b_columns[offset];
              b_values[offset - correction] = b_values[offset];
            }

            partitions[corrected_column] -= correction;
          }

          partitions[n_columns] -= correction;
        }
      }

      index_type new_nnz_without_duplicates = partitions[n_columns];
      B->setSymmetric(A->symmetric());
      B->setNnz(new_nnz_without_duplicates);
      // NOTE: see the comment in coo2csr for why this is necessary
      B->setNnzExpanded(new_nnz_without_duplicates);
      B->setExpanded(true);

      if (B->updateData(b_rows, b_columns, b_values, memory::HOST, memspace) != 0) {
        delete[] b_rows;
        delete[] b_columns;
        delete[] b_values;

        assert(false && "invalid state after coo -> coo conversion");
        return -1;
      }

      delete[] b_rows;
      delete[] b_columns;
      delete[] b_values;

      return 0;
    }

    /**
     * @param[in]  A_coo
     * @param[out] A_csr
     * @param[in] memspace
     * @return int - Error code, 0 if successful.
     *
     * @pre `A_coo` is a valid sparse matrix in unordered COO format. Duplicates are
     * allowed. Up-to-date values and indices must be on the host.
     *
     * @post `A_csr` represents the same matrix as `A_coo` but is in the CSR format.
     * `A_csr` is allocated and stored on the host.
     *
     * @invariant `A_coo` is not changed.
     */
    int coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace)
    {
      index_type* coo_rows = A_coo->getRowData(memory::HOST);
      index_type* coo_columns = A_coo->getColData(memory::HOST);
      real_type* coo_values = A_coo->getValues(memory::HOST);

      if (coo_rows == nullptr || coo_columns == nullptr || coo_values == nullptr) {
        return 0;
      }

      index_type nnz_with_duplicates = A_coo->getNnz();
      index_type n_rows = A_coo->getNumRows();
      index_type* csr_rows = new index_type[n_rows + 1];
      std::fill_n(csr_rows, n_rows + 1, 0);

      // NOTE: this is the only auxiliary storage buffer used by this conversion
      //       function. it is first used to track the number of values off the
      //       diagonal (if the matrix is symmetric and unexpanded), then it is
      //       used to track the amount of elements present within each row while
      //       the rows are being filled. this is later used during the backshifting
      //       step, in which the excess space is compacted so that there is no
      //       left over space between each row
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

      if (!A_coo->symmetric() || A_coo->expanded()) {
        for (index_type i = 0; i < nnz_with_duplicates; i++) {
          csr_rows[coo_rows[i] + 1]++;
        }
      } else {
        bool is_upper_triangular = false;
        bool is_lower_triangular = false;
        for (index_type i = 0; i < nnz_with_duplicates; i++) {
          csr_rows[coo_rows[i] + 1]++;
          if (coo_rows[i] != coo_columns[i]) {
            used[coo_columns[i]]++;

            is_upper_triangular |= coo_rows[i] < coo_columns[i];
            is_lower_triangular |= coo_rows[i] > coo_columns[i];
            if (is_upper_triangular && is_lower_triangular) {
              assert(false && "a matrix indicated to be symmetric triangular was not actually symmetric triangular");
              return -1;
            }
          }
        }
      }

      for (index_type row = 0; row < n_rows; row++) {
        csr_rows[row + 1] += csr_rows[row] + used[row];
        used[row] = 0;
      }

      index_type new_nnz_with_duplicates = csr_rows[n_rows];
      index_type* csr_columns = new index_type[new_nnz_with_duplicates];
      std::fill_n(csr_columns, new_nnz_with_duplicates, -1);
      real_type* csr_values = new real_type[new_nnz_with_duplicates];

      // fill stage, approximately O(nnz * m) in the worst case
      //
      // all this does is iterate over the nonzeroes in the coo matrix,
      // check to see if a value at that column already exists using binary search,
      // and if it does, then add to the value at that position, deduplicating
      // the matrix. otherwise, it allocates a new spot in the row (where you see
      // used[coo_rows[i]]++) and shifts everything over, performing what is
      // effectively insertion sort. the lower half is conditioned on the matrix
      // being symmetric and stored as either upper-triangular or lower-triangular,
      // and just performs the same as what is described above, but with the
      // indices swapped.

      for (index_type i = 0; i < nnz_with_duplicates; i++) {
        // this points to the first element not less than `coo_columns[i]`. see
        // https://en.cppreference.com/w/cpp/algorithm/lower_bound for more details
        index_type* closest_position =
            std::lower_bound(&csr_columns[csr_rows[coo_rows[i]]],
                             &csr_columns[csr_rows[coo_rows[i]] + used[coo_rows[i]]],
                             coo_columns[i]);

        // this is the offset at which the element's value belongs
        index_type insertion_offset = static_cast<index_type>(closest_position - csr_columns);

        if (csr_columns[insertion_offset] == coo_columns[i]) {
          csr_values[insertion_offset] += coo_values[i];
        } else {
          for (index_type offset = csr_rows[coo_rows[i]] + used[coo_rows[i]]++;
               offset > insertion_offset;
               offset--) {
            std::swap(csr_columns[offset], csr_columns[offset - 1]);
            std::swap(csr_values[offset], csr_values[offset - 1]);
          }

          csr_columns[insertion_offset] = coo_columns[i];
          csr_values[insertion_offset] = coo_values[i];
        }

        if ((A_coo->symmetric() && !A_coo->expanded()) && (coo_columns[i] != coo_rows[i])) {
          index_type* mirrored_closest_position =
              std::lower_bound(&csr_columns[csr_rows[coo_columns[i]]],
                               &csr_columns[csr_rows[coo_columns[i]] + used[coo_columns[i]]],
                               coo_rows[i]);
          index_type mirrored_insertion_offset = static_cast<index_type>(mirrored_closest_position - csr_columns);

          if (csr_columns[mirrored_insertion_offset] == coo_rows[i]) {
            csr_values[mirrored_insertion_offset] += coo_values[i];
          } else {
            for (index_type offset = csr_rows[coo_columns[i]] + used[coo_columns[i]]++;
                 offset > mirrored_insertion_offset;
                 offset--) {
              std::swap(csr_columns[offset], csr_columns[offset - 1]);
              std::swap(csr_values[offset], csr_values[offset - 1]);
            }

            csr_columns[mirrored_insertion_offset] = coo_rows[i];
            csr_values[mirrored_insertion_offset] = coo_values[i];
          }
        }
      }

      // backshifting stage, approximately O(nnz * m) in the worst case
      //
      // all this does is shift back rows to remove empty space in between them
      // by indexing each row in order, checking to see if the amount of used
      // spaces is equivalent to the amount of nonzeroes in the row, and if not,
      // shifts every subsequent row back the amount of unused spaces

      for (index_type row = 0; row < n_rows; row++) {
        index_type row_nnz = csr_rows[row + 1] - csr_rows[row];
        if (used[row] != row_nnz) {
          index_type correction = row_nnz - used[row];

          for (index_type corrected_row = row + 1;
               corrected_row < n_rows;
               corrected_row++) {
            for (index_type offset = csr_rows[corrected_row];
                 offset < csr_rows[corrected_row + 1];
                 offset++) {
              csr_columns[offset - correction] = csr_columns[offset];
              csr_values[offset - correction] = csr_values[offset];
            }

            csr_rows[corrected_row] -= correction;
          }

          csr_rows[n_rows] -= correction;
        }
      }

      index_type new_nnz_without_duplicates = csr_rows[n_rows];
      A_csr->setSymmetric(A_coo->symmetric());
      A_csr->setNnz(new_nnz_without_duplicates);
      // NOTE: this is necessary because updateData always reads the current nnz from
      //       this field. see #176
      A_csr->setNnzExpanded(new_nnz_without_duplicates);
      A_csr->setExpanded(true);

      if (A_csr->updateData(csr_rows, csr_columns, csr_values, memory::HOST, memspace) != 0) {
        delete[] csr_rows;
        delete[] csr_columns;
        delete[] csr_values;

        assert(false && "invalid state after coo -> csr conversion");
        return -1;
      }

      delete[] csr_rows;
      delete[] csr_columns;
      delete[] csr_values;

      return 0;
    }
  } // namespace matrix
} // namespace ReSolve
