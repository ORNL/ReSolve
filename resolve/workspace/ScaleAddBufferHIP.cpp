#include <resolve/workspace/ScaleAddBufferHIP.hpp>

#include <cassert>

namespace ReSolve
{
  /**
   * @brief Store sparsity pattern
   *
   * @param[in] row_data - pointer to row data (array of integers, length:nrows+1)
   * @param[in] nrows - number of rows
   * @param[in] col_data - pointer to column data (array of integers, length: nnz)
   * @param[in] nnz - number of non-zeros
   */
  ScaleAddBufferHIP::ScaleAddBufferHIP(index_type numRows)
    : numRows_(numRows)
  {
    rocsparse_create_mat_descr(&mat_A_);
    mem_.allocateArrayOnDevice(&rowData_, numRows_ + 1);
  }

  /**
   * @brief Destructor
   *
   */
  ScaleAddBufferHIP::~ScaleAddBufferHIP()
  {
    mem_.deleteOnDevice(rowData_);
    rocsparse_destroy_mat_descr(mat_A_);
  }

  /**
   * @brief Retrieve row sparsity pattern
   *
   * @return precalculated row pointers
   */
  index_type* ScaleAddBufferHIP::getRowData()
  {
    return rowData_;
  }

  /**
   * @brief Retrieve matrix descriptor
   *
   * @return matrix descriptor set for scaleAddB, scaleAddI
   */
  rocsparse_mat_descr ScaleAddBufferHIP::getMatrixDescriptor()
  {
    return mat_A_;
  }

  /**
   * @brief get number of matrix rows
   *
   * @return number of matrix rows.
   */
  index_type ScaleAddBufferHIP::getNumRows()
  {
    return numRows_;
  }

  /**
   * @brief Get number of non-zeros.
   *
   * @return number of non-zeros
   */
  index_type ScaleAddBufferHIP::getNnz()
  {
    return nnz_;
  }

  /**
   * @brief Get number of non-zeros.
   *
   * @param[in] nnz number of non-zeros
   */
  void ScaleAddBufferHIP::setNnz(index_type nnz)
  {
    nnz_ = nnz;
  }
}
