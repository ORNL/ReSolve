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
  ScaleAddIBuffer::ScaleAddIBuffer(index_type* row_data, index_type nrows, index_type* col_data, index_type nnz)
    : row_data_(row_data, row_data + nrows + 1), col_data_(col_data, col_data + nnz)
  {
  }

  index_type* ScaleAddIBuffer::getRowData()
  {
    return row_data_.data();
  }

  index_type* ScaleAddIBuffer::getColumnData()
  {
    return col_data_.data();
  }

  /**
   * @brief get number of matrix rows
   *
   * @return number of matrix rows.
   */
  index_type ScaleAddIBuffer::getNumRows()
  {
    return static_cast<index_type>(row_data_.size()) - 1;
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
    return static_cast<index_type>(col_data_.size());
  }

  LinAlgWorkspaceCpu::LinAlgWorkspaceCpu()
  {
  }

  LinAlgWorkspaceCpu::~LinAlgWorkspaceCpu()
  {
    delete scaleaddi_buffer_;
  }

  void LinAlgWorkspaceCpu::initializeHandles()
  {
  }

  void LinAlgWorkspaceCpu::resetLinAlgWorkspace()
  {
    delete scaleaddi_buffer_;
    scaleaddi_buffer_ = nullptr;
  }

  bool LinAlgWorkspaceCpu::scaleAddISetup()
  {
    return scaleaddi_setup_done_;
  }

  void LinAlgWorkspaceCpu::scaleAddISetupDone()
  {
    scaleaddi_setup_done_ = true;
  }

  ScaleAddIBuffer* LinAlgWorkspaceCpu::getScaleAddIBuffer()
  {
    assert(scaleaddi_buffer_ != nullptr);
    return scaleaddi_buffer_;
  }

  void LinAlgWorkspaceCpu::setScaleAddIBuffer(ScaleAddIBuffer* buffer)
  {
    assert(scaleaddi_buffer_ == nullptr);
    scaleaddi_buffer_ = buffer;
  }

} // namespace ReSolve
