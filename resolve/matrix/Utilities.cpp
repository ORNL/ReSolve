#include <algorithm>
#include <resolve/Common.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include "Utilities.hpp"

namespace ReSolve
{
  /// @brief Helper class for COO matrix sorting
  class IndexValuePair
  {
    public:
      IndexValuePair() : idx_(0), value_(0.0)
      {}
      ~IndexValuePair()
      {}
      void setIdx (index_type new_idx)
      {
        idx_ = new_idx;
      }
      void setValue (real_type new_value)
      {
        value_ = new_value;
      }

      index_type getIdx()
      {
        return idx_;
      }
      real_type getValue()
      {
        return value_;
      }

      bool operator < (const IndexValuePair& str) const
      {
        return (idx_ < str.idx_);
      }  

    private:
      index_type idx_;
      real_type value_;
  };

  using out = io::Logger;
  namespace matrix
  {
    /**
     * @brief Creates a CSR from a COO matrix.
     * 
     * @param[in]  A_coo 
     * @param[out] A_csr 
     * @return int - Error code, 0 if successful.
     * 
     * @pre `A_coo` is a valid sparse matrix in unordered COO format.
     * Duplicates are allowed. Up-to-date values and indices must be
     * on host.
     * 
     * @post `A_csr` is representing the same matrix as `A_coo` but in
     * _general_ CSR format. `A_csr` is allocated and stored on host.
     * 
     * @invariant `A_coo` is not changed.
     */
    int coo2csr(matrix::Coo* A_coo, matrix::Csr* A_csr, memory::MemorySpace memspace)
    {
      //count nnzs first
      index_type nnz_unpacked = 0;
      const index_type nnz = A_coo->getNnz();
      const index_type n = A_coo->getNumRows();
      const bool symmetric = A_coo->symmetric();
      const bool expanded = A_coo->expanded();

      index_type* nnz_counts = new index_type[n]; ///< Number of elements per row
      std::fill_n(nnz_counts, n, 0);
      const index_type* coo_rows = A_coo->getRowData(memory::HOST);
      const index_type* coo_cols = A_coo->getColData(memory::HOST);
      const real_type*  coo_vals = A_coo->getValues( memory::HOST);

      index_type* diag_control = new index_type[n]; // for DEDUPLICATION of the diagonal
      std::fill_n(diag_control, n, 0);
      index_type nnz_unpacked_no_duplicates = 0;
      index_type nnz_no_duplicates = nnz;

      // Count matrix elements
      for (index_type i = 0; i < nnz; ++i) {
        nnz_counts[coo_rows[i]]++;
        nnz_unpacked++;
        nnz_unpacked_no_duplicates++;
        // Count elements added after expansion
        if ((coo_rows[i] != coo_cols[i]) && (symmetric) && (!expanded)) {
          nnz_counts[coo_cols[i]]++;
          nnz_unpacked++;
          nnz_unpacked_no_duplicates++;
        }
        // Bookkeeping of diagonal elements that were counted
        if (coo_rows[i] == coo_cols[i]) {
          if (diag_control[coo_rows[i]] > 0) {
            // duplicate
            nnz_unpacked_no_duplicates--;
            nnz_no_duplicates--;
          }
          diag_control[coo_rows[i]]++;
        }
      }
      A_csr->setExpanded(true);
      A_csr->setNnzExpanded(nnz_unpacked_no_duplicates);

      // Allocate matrix format conversion workspace
      index_type* csr_ia = new index_type[n+1];
      std::fill_n(csr_ia, n + 1, 0);
      index_type* csr_ja = new index_type[nnz_unpacked];
      real_type* csr_a = new real_type[nnz_unpacked];
      index_type* nnz_shifts = new index_type[n];
      std::fill_n(nnz_shifts, n , 0);

      IndexValuePair* tmp = new IndexValuePair[nnz_unpacked]; 

      // Set CSR row pointers
      csr_ia[0] = 0;
      for (index_type i = 1; i < n + 1; ++i) {
        csr_ia[i] = csr_ia[i - 1] + nnz_counts[i - 1] - (diag_control[i-1] - 1);
      }

      int r, start;

      for (index_type i = 0; i < nnz; ++i) {
        //which row
        r = coo_rows[i];
        start = csr_ia[r];

        if ((start + nnz_shifts[r]) > nnz_unpacked) {
          out::warning() << "index out of bounds (case 1) start: " << start 
                         << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
        }
        if ((r == coo_cols[i]) && (diag_control[r] > 1)) {//diagonal, and there are duplicates
          bool already_there = false;  
          for (index_type j = start; j < start + nnz_shifts[r]; ++j)
          {
            index_type c = tmp[j].getIdx();
            if (c == r) {
              real_type val = tmp[j].getValue();
              val += coo_vals[i];
              tmp[j].setValue(val);
              already_there = true;
              out::warning() << " duplicate found, row " << c 
                             << " adding in place " << j << " current value: " << val << std::endl;
            }  
          }  
          if (!already_there) { // first time this duplicates appears

            tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
            tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);

            nnz_shifts[r]++;
          }
        } else { // non-diagonal
          tmp[start + nnz_shifts[r]].setIdx(coo_cols[i]);
          tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
          nnz_shifts[r]++;

          if ((coo_rows[i] != coo_cols[i]) && (symmetric == 1)) {
            r = coo_cols[i];
            start = csr_ia[r];

            if ((start + nnz_shifts[r]) > nnz_unpacked) {
              out::warning() << "index out of bounds (case 2) start: " << start 
                             << "nnz_shifts[" << r << "] = " << nnz_shifts[r] << std::endl;
            }
            tmp[start + nnz_shifts[r]].setIdx(coo_rows[i]);
            tmp[start + nnz_shifts[r]].setValue(coo_vals[i]);
            nnz_shifts[r]++;
          }
        }
      }

      //now sort whatever is inside rows
      for (int i = 0; i < n; ++i) {
        //now sorting (and adding 1)
        int colStart = csr_ia[i];
        int colEnd = csr_ia[i + 1];
        int length = colEnd - colStart;
        std::sort(&tmp[colStart], &tmp[colStart] + length);
      }

      for (index_type i = 0; i < nnz_unpacked; ++i)
      {
        csr_ja[i] = tmp[i].getIdx();
        csr_a[i] = tmp[i].getValue();
      }
      A_csr->setNnz(nnz_no_duplicates);
      A_csr->updateData(csr_ia, csr_ja, csr_a, memory::HOST, memspace);

      delete [] nnz_counts;
      delete [] tmp;
      delete [] nnz_shifts;
      delete [] csr_ia;
      delete [] csr_ja;
      delete [] csr_a;
      delete [] diag_control; 

      return 0;

    }
  }
}