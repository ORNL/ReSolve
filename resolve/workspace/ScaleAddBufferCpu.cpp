#include <algorithm>
#include <cassert>
#include <cstddef>

#include <resolve/workspace/ScaleAddBufferCpu.hpp>

namespace ReSolve
{
  /**
   * @brief Store sparsity pattern
   *
   * @param[in] row_data - row data (array of integers, length:nrows+1)
   * @param[in] col_data - column data (array of integers, length: nnz)
   */
  ScaleAddBufferCpu::ScaleAddBufferCpu(std::vector<index_type> rowData, std::vector<index_type> colData)
    : rowData_(std::move(rowData)), colData_(std::move(colData))
  {
  }

  /**
   * @brief Retrieve row sparsity pattern
   *
   * @return precalculated row pointers
   */
  index_type* ScaleAddBufferCpu::getRowData()
  {
    return rowData_.data();
  }

  /**
   * @brief Retrieve column sparsity pattern
   *
   * @return precalculated column indices
   */
  index_type* ScaleAddBufferCpu::getColumnData()
  {
    return colData_.data();
  }

  /**
   * @brief get number of matrix rows
   *
   * @return number of matrix rows.
   */
  index_type ScaleAddBufferCpu::getNumRows()
  {
    return static_cast<index_type>(rowData_.size()) - 1;
  }

  /**
   * @brief get number of matrix columns
   *
   * @return number of matrix columns.
   */
  index_type ScaleAddBufferCpu::getNumColumns()
  {
    return getNumRows();
  }

  /**
   * @brief Get number of non-zeros.
   *
   * @return number of non-zeros
   */
  index_type ScaleAddBufferCpu::getNnz()
  {
    return static_cast<index_type>(colData_.size());
  }
} // namespace ReSolve
