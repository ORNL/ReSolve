// Matrix utilities
// Mirroring memory approach
#pragma once
#include <string>
#include <functional>
#include <tuple>
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve { namespace matrix {
  /**
   * @brief This class implements basic sparse matrix interface. (Almost) all sparse matrix formats store information about matrix rows and columns (as integers) and data (as real numbers).
   *        This class is virtualand implements only what is common for all basic formats.
   *        Note that regardless of how row/column information is stored, all values need to be stored, so all utilities needed for values are implemented in this class.
   *
   * @author Kasia Swirydowicz <kasia.swirydowicz@pnnl.gov>
   */
  class Sparse
  {
    public:
      //basic constructor
      Sparse();
      Sparse(index_type n, index_type m, index_type nnz);
      Sparse(index_type n,
             index_type m,
             index_type nnz,
             bool symmetric,
             bool expanded);
      virtual ~Sparse();

      // accessors
      index_type getNumRows();
      index_type getNumColumns();
      index_type getNnz();
      index_type getNnzExpanded();

      bool symmetric();
      bool expanded();
      void setSymmetric(bool symmetric);
      void setExpanded(bool expanded);
      void setNnzExpanded(index_type nnz_expanded_new);
      void setNnz(index_type nnz_new); // for resetting when removing duplicates
      int setUpdated(memory::MemorySpace what);

      virtual index_type* getRowData(memory::MemorySpace memspace) = 0;
      virtual index_type* getColData(memory::MemorySpace memspace) = 0;
      virtual real_type*  getValues( memory::MemorySpace memspace) = 0;

      virtual int updateData(index_type* row_data, index_type* col_data, real_type* val_data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut) = 0;
      virtual int updateData(index_type* row_data, index_type* col_data, real_type* val_data, index_type new_nnz, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut) = 0;

      virtual int allocateMatrixData(memory::MemorySpace memspace) = 0;
      int setMatrixData(index_type* row_data, index_type* col_data, real_type* val_data, memory::MemorySpace memspace);

      int destroyMatrixData(memory::MemorySpace memspace);

      virtual void print(std::ostream& file_out) = 0;

      virtual int copyData(memory::MemorySpace memspaceOut) = 0;

      int setDataOwnership(bool, memory::MemorySpace);
      int setValueOwnership(bool, memory::MemorySpace);

      //update Values just updates values; it allocates if necessary.
      //values have the same dimensions between different formats
      virtual int updateValues(real_type* new_vals, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut);

      //set new values just sets the pointer, use caution.
      virtual int setNewValues(real_type* new_vals, memory::MemorySpace memspace);

    protected:
      //size
      index_type n_{0}; ///< number of rows
      index_type m_{0}; ///< number of columns
      index_type nnz_{0}; ///< number of non-zeros
      index_type nnz_expanded_{0}; ///< number of non-zeros in an expanded matrix (for symmetric matrices)

      bool is_symmetric_{false}; ///< symmetry flag
      bool is_expanded_{false}; ///< "expanded" flag

      //host data
      index_type* h_row_data_{nullptr}; ///< row data (HOST)
      index_type* h_col_data_{nullptr}; ///< column data (HOST)
      real_type* h_val_data_{nullptr}; ///< value data (HOST)
      bool h_data_updated_{false}; ///< HOST update flag

      //gpu data
      index_type* d_row_data_{nullptr}; ///< row data (DEVICE)
      index_type* d_col_data_{nullptr}; ///< column data (DEVICE)
      real_type* d_val_data_{nullptr}; ///< value data (DEVICE)
      bool d_data_updated_{false}; ///< DEVICE update flag

      void setNotUpdated();

      // Data ownership flags
      bool owns_cpu_data_{false}; ///< for row/col data
      bool owns_cpu_vals_{false}; ///< for values

      bool owns_gpu_data_{false}; ///< for row/col data
      bool owns_gpu_vals_{false}; ///< for values

      MemoryHandler mem_; ///< Device memory manager object
  };
}} // namespace ReSolve::matrix
