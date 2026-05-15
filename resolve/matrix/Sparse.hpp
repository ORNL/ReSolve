// Matrix utilities
// Mirroring memory approach
#pragma once
#include <functional>
#include <string>
#include <tuple>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Matrix.hpp>

namespace ReSolve
{
  namespace matrix
  {

    /**
     * @brief This class implements basic sparse matrix interface.
     *
     * Most of sparse matrix formats store information about matrix rows and
     * columns as integers and nonzero element values as real numbers.
     * This class is virtual and implements only what is common for all basic
     * formats. Note that regardless of how row/column information is stored,
     * all nonzero matrix values need to be stored, so all utilities needed for
     * the values are implemented in this class.
     *
     * @author Kasia Swirydowicz <kasia.swirydowicz@pnnl.gov>
     */
    class Sparse : public Matrix
    {
    public:
      /// Supported sparse matrix formats
      enum SparseFormat
      {
        NONE,
        TRIPLET,
        COMPRESSED_SPARSE_ROW,
        COMPRESSED_SPARSE_COLUMN
      };

    public:
      // basic constructor
      Sparse();
      Sparse(index_type n, index_type m, index_type nnz);
      Sparse(index_type n,
             index_type m,
             index_type nnz,
             bool       symmetric,
             bool       expanded);
      virtual ~Sparse();

      SparseFormat getSparseFormat() const;

      bool symmetric();
      bool expanded();
      void setSymmetric(bool symmetric);
      void setExpanded(bool expanded);
      int  setUpdated(memory::MemorySpace what);

      virtual index_type* getRowData(memory::MemorySpace memspace) = 0;
      virtual index_type* getColData(memory::MemorySpace memspace) = 0;
      virtual real_type*  getValues(memory::MemorySpace memspace)  = 0;

      virtual int copyFromExternal(const index_type*   row_data,
                                   const index_type*   col_data,
                                   const real_type*    val_data,
                                   memory::MemorySpace memspaceIn,
                                   memory::MemorySpace memspaceOut) = 0;
      virtual int copyFromExternal(const index_type*   row_data,
                                   const index_type*   col_data,
                                   const real_type*    val_data,
                                   index_type          new_nnz,
                                   memory::MemorySpace memspaceIn,
                                   memory::MemorySpace memspaceOut) = 0;

      virtual int allocateMatrixData(memory::MemorySpace memspace) = 0;
      int         setDataPointers(index_type*         row_data,
                                  index_type*         col_data,
                                  real_type*          val_data,
                                  memory::MemorySpace memspace);

      int destroyMatrixData(memory::MemorySpace memspace);

      virtual int syncData(memory::MemorySpace memspaceOut) = 0;

      // update Values just updates values; it allocates if necessary.
      // values have the same dimensions between different formats
      virtual int copyValues(const real_type*    new_vals,
                             memory::MemorySpace memspaceIn,
                             memory::MemorySpace memspaceOut);

      // set new values just sets the pointer, use caution.
      virtual int setValuesPointer(real_type*          new_vals,
                                   memory::MemorySpace memspace);

      // matrix operations
      virtual int addConst(real_type alpha, memory::MemorySpace memspace);
      virtual int matvec(
          vector_type*        vec_x,
          vector_type*        vec_result,
          const real_type*    alpha,
          const real_type*    beta,
          memory::MemorySpace memspace);
      virtual int matrixInfNorm(real_type* norm, memory::MemorySpace memspace);

    protected:
      SparseFormat sparse_format_{NONE}; ///< Matrix format

      bool is_symmetric_{false}; ///< symmetry flag
      bool is_expanded_{false};  ///< "expanded" flag

      // host data
      index_type* h_row_data_{nullptr};   ///< row data (HOST)
      index_type* h_col_data_{nullptr};   ///< column data (HOST)
      real_type*  h_val_data_{nullptr};   ///< value data (HOST)
      bool        h_data_updated_{false}; ///< HOST update flag

      // gpu data
      index_type* d_row_data_{nullptr};   ///< row data (DEVICE)
      index_type* d_col_data_{nullptr};   ///< column data (DEVICE)
      real_type*  d_val_data_{nullptr};   ///< value data (DEVICE)
      bool        d_data_updated_{false}; ///< DEVICE update flag

      void setNotUpdated();

      // Data ownership flags
      bool owns_cpu_sparsity_pattern_{false}; ///< for row/col data
      bool owns_cpu_values_{false};           ///< for nonzero values

      bool owns_gpu_sparsity_pattern_{false}; ///< for row/col data
      bool owns_gpu_values_{false};           ///< for nonzero values

      MemoryHandler mem_; ///< Device memory manager object
    };
  } // namespace matrix
} // namespace ReSolve
