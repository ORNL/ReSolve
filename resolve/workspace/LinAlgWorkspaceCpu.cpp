#include "LinAlgWorkspaceCpu.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>

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
  ScaleAddIBuffer::ScaleAddIBuffer(std::vector<index_type> rowData, std::vector<index_type> colData)
    : rowData_(std::move(rowData)), colData_(std::move(colData))
  {
  }

  /**
   * @brief Retrieve row sparsity pattern
   *
   * @return precalculated row pointers
   */
  index_type* ScaleAddIBuffer::getRowData()
  {
    return rowData_.data();
  }

  /**
   * @brief Retrieve column sparsity pattern
   *
   * @return precalculated column indices
   */
  index_type* ScaleAddIBuffer::getColumnData()
  {
    return colData_.data();
  }

  /**
   * @brief get number of matrix rows
   *
   * @return number of matrix rows.
   */
  index_type ScaleAddIBuffer::getNumRows()
  {
    return static_cast<index_type>(rowData_.size()) - 1;
  }

  /**
   * @brief get number of matrix columns
   *
   * @return number of matrix columns.
   */
  index_type ScaleAddIBuffer::getNumColumns()
  {
    return getNumRows();
  }

  /**
   * @brief Get number of non-zeros.
   *
   * @return number of non-zeros
   */
  index_type ScaleAddIBuffer::getNnz()
  {
    return static_cast<index_type>(colData_.size());
  }

  LinAlgWorkspaceCpu::LinAlgWorkspaceCpu()
  {
  }

  LinAlgWorkspaceCpu::~LinAlgWorkspaceCpu()
  {
    delete scaleAddIBuffer_;
  }

  void LinAlgWorkspaceCpu::initializeHandles()
  {
  }

  void LinAlgWorkspaceCpu::resetLinAlgWorkspace()
  {
    delete scaleAddIBuffer_;
    scaleAddIBuffer_ = nullptr;
  }

  bool LinAlgWorkspaceCpu::scaleAddISetup()
  {
    return scaleAddISetupDone_;
  }

  void LinAlgWorkspaceCpu::scaleAddISetupDone()
  {
    scaleAddISetupDone_ = true;
  }

  ScaleAddIBuffer* LinAlgWorkspaceCpu::getScaleAddIBuffer()
  {
    assert(scaleAddIBuffer_ != nullptr);
    return scaleAddIBuffer_;
  }

  void LinAlgWorkspaceCpu::setScaleAddIBuffer(ScaleAddIBuffer* buffer)
  {
    assert(scaleAddIBuffer_ == nullptr);
    scaleAddIBuffer_ = buffer;
    scaleAddISetupDone();
  }

} // namespace ReSolve
