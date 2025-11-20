#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>

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
  ScaleAddBufferHIP::ScaleAddBufferHIP(index_type numRows, size_t bufferSize)
    : numRows_(numRows), bufferSize_(bufferSize)
  {
    mem_.allocateArrayOnDevice(&rowData_, numRows_ + 1);
    mem_.allocateBufferOnDevice(&buffer_, bufferSize_);
  }

  /**
   * @brief Destructor
   *
   */
  ScaleAddBufferHIP::~ScaleAddBufferHIP()
  {
    mem_.deleteOnDevice(rowData_);
    mem_.deleteOnDevice(buffer_);
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
   * @brief Retrieve row sparsity pattern
   *
   * @return precalculated row pointers
   */
  void* ScaleAddBufferHIP::getBuffer()
  {
    return buffer_;
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
   * @return number of non-zeros
   */
  void ScaleAddBufferHIP::setNnz(index_type nnz)
  {
    nnz_ = nnz;
  }

  LinAlgWorkspaceHIP::LinAlgWorkspaceHIP()
  {
    handle_rocsparse_ = nullptr;
    handle_rocblas_   = nullptr;
    buffer_scale_add_i_        = nullptr;
    buffer_scale_add_b_        = nullptr;
    matvec_setup_done_         = false;
    d_r_                       = nullptr;
    d_r_size_                  = 0;
    norm_buffer_               = nullptr;
    norm_buffer_ready_         = false;
    transpose_workspace_       = nullptr;
    transpose_workspace_ready_ = false;
  }

  LinAlgWorkspaceHIP::~LinAlgWorkspaceHIP()
  {
    rocsparse_destroy_handle(handle_rocsparse_);
    rocblas_destroy_handle(handle_rocblas_);
    if (matvec_setup_done_)
    {
      rocsparse_destroy_mat_descr(mat_A_);
    }
    if (scale_add_i_setup_done_)
    {
      assert(buffer_scale_add_i_ != nullptr);
      delete buffer_scale_add_i_;
      rocsparse_destroy_mat_descr(mat_B_);
    }
    if (buffer_scale_add_b_ != nullptr)
    {
      assert(buffer_scale_add_b_ != nullptr);
      delete buffer_scale_add_b_;
      rocsparse_destroy_mat_descr(mat_B_);
    }
    if (d_r_size_ != 0)
    {
      mem_.deleteOnDevice(d_r_);
    }
    if (norm_buffer_ready_ == true)
    {
      mem_.deleteOnDevice(norm_buffer_);
    }
    if (transpose_workspace_ready_)
    {
      mem_.deleteOnDevice(transpose_workspace_);
    }
  }

  /**
   * @brief Resets the linear algebra workspace.
   *
   * This function clears the state of the linear algebra workspace by
   * destroying the matrix descriptor, deallocating the residual vector,
   * deleting the norm buffer, and resetting the transpose workspace.
   */
  void LinAlgWorkspaceHIP::resetLinAlgWorkspace()
  {
    if (matvec_setup_done_)
    {
      rocsparse_destroy_mat_descr(mat_A_);
      matvec_setup_done_ = false;
    }
    if (scale_add_b_setup_done_)
    {
      delete buffer_scale_add_b_;
      buffer_scale_add_b_     = nullptr;
      rocsparse_destroy_mat_descr(mat_B_);
      scale_add_b_setup_done_ = false;
    }
    if (scale_add_i_setup_done_)
    {
      delete buffer_scale_add_i_;
      buffer_scale_add_i_     = nullptr;
      scale_add_i_setup_done_ = false;
    }
    if (d_r_size_ != 0)
    {
      mem_.deleteOnDevice(d_r_);
      d_r_      = nullptr;
      d_r_size_ = 0;
    }
    if (norm_buffer_ready_ == true)
    {
      mem_.deleteOnDevice(norm_buffer_);
      norm_buffer_       = nullptr;
      norm_buffer_ready_ = false;
    }
    if (transpose_workspace_ready_)
    {
      mem_.deleteOnDevice(transpose_workspace_);
      transpose_workspace_       = nullptr;
      transpose_workspace_ready_ = false;
    }
    return;
  }

  rocsparse_handle LinAlgWorkspaceHIP::getRocsparseHandle()
  {
    return handle_rocsparse_;
  }

  void LinAlgWorkspaceHIP::setRocsparseHandle(rocsparse_handle handle)
  {
    handle_rocsparse_ = handle;
  }

  rocblas_handle LinAlgWorkspaceHIP::getRocblasHandle()
  {
    return handle_rocblas_;
  }

  void LinAlgWorkspaceHIP::setRocblasHandle(rocblas_handle handle)
  {
    handle_rocblas_ = handle;
  }

  rocsparse_mat_descr LinAlgWorkspaceHIP::getSpmvMatrixDescriptor()
  {
    return mat_A_;
  }

  rocsparse_mat_descr LinAlgWorkspaceHIP::getScaleAddMatrixDescriptor()
  {
    return mat_B_;
  }

  void LinAlgWorkspaceHIP::setSpmvMatrixDescriptor(rocsparse_mat_descr mat)
  {
    mat_A_ = mat;
  }

  void LinAlgWorkspaceHIP::setScaleAddMatrixDescriptor(rocsparse_mat_descr mat)
  {
    mat_B_ = mat;
  }

  rocsparse_mat_info LinAlgWorkspaceHIP::getSpmvMatrixInfo()
  {
    return info_A_;
  }

  void LinAlgWorkspaceHIP::setSpmvMatrixInfo(rocsparse_mat_info info)
  {
    info_A_ = info;
  }

  void LinAlgWorkspaceHIP::setDrSize(index_type new_sz)
  {
    d_r_size_ = new_sz;
  }

  void LinAlgWorkspaceHIP::setDr(double* new_dr)
  {
    d_r_ = new_dr;
  }

  void LinAlgWorkspaceHIP::setNormBuffer(double* nb)
  {
    norm_buffer_ = nb;
  }

  ScaleAddBufferHIP* LinAlgWorkspaceHIP::getScaleAddIBuffer()
  {
    return buffer_scale_add_i_;
  }

  ScaleAddBufferHIP* LinAlgWorkspaceHIP::getScaleAddBBuffer()
  {
    return buffer_scale_add_b_;
  }

  void LinAlgWorkspaceHIP::setScaleAddBBuffer(ScaleAddBufferHIP* buffer)
  {
    buffer_scale_add_b_ = buffer;
  }

  void LinAlgWorkspaceHIP::setScaleAddIBuffer(ScaleAddBufferHIP* buffer)
  {
    buffer_scale_add_i_ = buffer;
  }

  void LinAlgWorkspaceHIP::scaleAddBSetupDone()
  {
    scale_add_b_setup_done_ = true;
  }

  void LinAlgWorkspaceHIP::scaleAddISetupDone()
  {
    scale_add_i_setup_done_ = true;
  }

  void LinAlgWorkspaceHIP::setNormBufferState(bool r)
  {
    norm_buffer_ready_ = r;
  }

  bool LinAlgWorkspaceHIP::matvecSetup()
  {
    return matvec_setup_done_;
  }

  void LinAlgWorkspaceHIP::matvecSetupDone()
  {
    matvec_setup_done_ = true;
  }

  bool LinAlgWorkspaceHIP::scaleAddISetup()
  {
    return scale_add_i_setup_done_;
  }

  bool LinAlgWorkspaceHIP::scaleAddBSetup()
  {
    return scale_add_b_setup_done_;
  }

  void LinAlgWorkspaceHIP::initializeHandles()
  {
    rocsparse_create_handle(&handle_rocsparse_);
    rocblas_create_handle(&handle_rocblas_);
  }

  index_type LinAlgWorkspaceHIP::getDrSize()
  {
    return d_r_size_;
  }

  real_type* LinAlgWorkspaceHIP::getDr()
  {
    return d_r_;
  }

  bool LinAlgWorkspaceHIP::getNormBufferState()
  {
    return norm_buffer_ready_;
  }

  real_type* LinAlgWorkspaceHIP::getNormBuffer()
  {
    return norm_buffer_;
  }

  void* LinAlgWorkspaceHIP::getTransposeBufferWorkspace()
  {
    return transpose_workspace_;
  }

  void LinAlgWorkspaceHIP::setTransposeBufferWorkspace(size_t bufferSize)
  {
    if (transpose_workspace_ready_)
    {
      mem_.deleteOnDevice(transpose_workspace_);
    }
    mem_.allocateBufferOnDevice(&transpose_workspace_, bufferSize);
    transpose_workspace_ready_ = true;
  }

  bool LinAlgWorkspaceHIP::isTransposeBufferAllocated()
  {
    return transpose_workspace_ready_;
  }
} // namespace ReSolve
