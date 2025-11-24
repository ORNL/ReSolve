#include <resolve/workspace/ScaleAddBufferCUDA.hpp>

namespace ReSolve
{

  /**
   * @brief Store sparsity pattern
   *
   * @param[in] nrows - number of rows
   */
  ScaleAddBufferCUDA::ScaleAddBufferCUDA(index_type numRows)
    : numRows_(numRows), bufferSize_(0)
  {
    mem_.allocateArrayOnDevice(&rowData_, numRows_ + 1);
    cusparseCreateMatDescr(&mat_A_);
    cusparseSetMatType(mat_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat_A_, CUSPARSE_INDEX_BASE_ZERO);
  }

  /**
   * @brief Destructor
   *
   */
  ScaleAddBufferCUDA::~ScaleAddBufferCUDA()
  {
    mem_.deleteOnDevice(rowData_);
    mem_.deleteOnDevice(buffer_);
    cusparseDestroyMatDescr(mat_A_);
  }

  /**
   * @brief Retrieve row sparsity pattern
   *
   * @return precalculated row pointers
   */
  index_type* ScaleAddBufferCUDA::getRowData()
  {
    return rowData_;
  }

  /**
   * @brief Retrieve matrix descriptor
   *
   * @return matrix descriptor set for scaleAddB, scaleAddI
   */
  cusparseMatDescr_t ScaleAddBufferCUDA::getMatrixDescriptor()
  {
    return mat_A_;
  }

  /**
   * @brief Allocate memory for cusparse buffer
   *
   * @param[in] bufferSize calculated array size
   */
  void ScaleAddBufferCUDA::allocateBuffer(size_t bufferSize)
  {
    bufferSize_ = bufferSize;
    mem_.allocateBufferOnDevice(&buffer_, bufferSize_);
  }

  /**
   * @brief Retrieve cusparse buffer
   *
   * @return cusparse buffer
   */
  void* ScaleAddBufferCUDA::getBuffer()
  {
    return buffer_;
  }

  /**
   * @brief get number of matrix rows
   *
   * @return number of matrix rows.
   */
  index_type ScaleAddBufferCUDA::getNumRows()
  {
    return numRows_;
  }

  /**
   * @brief Get number of non-zeros.
   *
   * @return number of non-zeros
   */
  index_type ScaleAddBufferCUDA::getNnz()
  {
    return nnz_;
  }

  /**
   * @brief set number of non-zeros.
   *
   * @param[in] nnz number of non-zeros
   */
  void ScaleAddBufferCUDA::setNnz(index_type nnz)
  {
    nnz_ = nnz;
  }

} // namespace ReSolve