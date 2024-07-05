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
      // NOTE: a note on deduplication:
      //       currently, this function allocates more memory than necessary to contain
      //       the matrix. this is because it's impossible to cleanly check the amount
      //       of space each row needs ahead of time without storing a full dense matrix
      //       with the dimensions of A
      //
      //       additionally, this necessitates that we shrink the rows to remove this
      //       unused space, which is costly and necessitates a lot of shifting
      //
      //       one potential solution to this that i thought of was to store an array of
      //       bloom filters (maybe u64s or u128s) of the size of the number of columns,
      //       each filter indicating if there is a value stored at that column in a row
      //
      //       if, during the preprocessing step, we encounter a potential duplicate, we
      //       backtrack to see if there actually is one. given that the number of
      //       duplicates is probably small, this shouldn't carry too much of a
      //       performance penalty
      //
      //       if performance becomes a problem, this could be coupled with arrays
      //       maintaining the last and/or first seen indices in the coo arrays at which
      //       a row had an associated value, to speed up the backtracking process
      //
      //       this would be applied during the first allocation phase when an element
      //       for a row is encountered, prior to the increment of its size in
      //       `new_rows`

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

      // allocation stage

      // NOTE: so aside from tracking the number of mirrored values for allocation
      //       purposes, we also have to compute the amount of nonzeroes in each row
      //       since we're dealing with a COO input matrix. we store these values in
      //       row + 1 slots in new_rows, to avoid allocating extra space :)

      if (!A->symmetric() || A->expanded()) {
        // NOTE: this branch is special cased because there is a bunch of extra
        //       bookkeeping done if a matrix is symmetric and unexpanded
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
      // TODO: we need to find a better way to fix this. what this fixes is
      //       that otherwise, the column values are all zero by default and
      //       this seems to mess with the way the insertion position is
      //       selected if the column of the pair we're looking to insert
      //       is zero
      std::fill_n(new_columns, new_rows[n_rows], -1);
      real_type* new_values = new real_type[new_rows[n_rows]];

      // fill stage

      for (index_type i = 0; i < nnz; i++) {
        index_type insertion_pos =
            static_cast<index_type>(
                std::lower_bound(&new_columns[new_rows[rows[i]]],
                                 &new_columns[new_rows[rows[i]] + used[rows[i]]],
                                 columns[i]) -
                new_columns);

        // NOTE: the stuff beyond here is basically insertion sort. i'm not
        //       sure if it's the best way of going about sorting the
        //       column-value pairs, but it seemed to me to be the most
        //       natural way of going about this. the only potential
        //       benefit to using something else (that i can forsee) would
        //       be that on really large matrices with many nonzeroes in a
        //       row, something like mergesort might be better or maybe
        //       a reimpl of plain std::sort. the advantage offered by
        //       insertion sort here is that we can do it while we fill
        //       the row, as opposed to doing sorting in a postprocessing
        //       step as was done prior

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
                                   rows[i]) -
                  new_columns);

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

      // backshifting stage

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

      if (B->destroyMatrixData(memory::HOST) != 0 ||
          B->setMatrixData(new_rows, new_columns, new_values, memory::HOST) != 0) {
        assert(false && "invalid state after coo -> csr conversion");
        return -1;
      }

      // TODO: set data ownership / value ownership here. we'd be leaking
      //       memory here otherwise

      B->setSymmetric(A->symmetric());
      B->setNnz(new_rows[n_rows]);
      B->setExpanded(true);

      return 0;
    }
  } // namespace matrix
} // namespace ReSolve
