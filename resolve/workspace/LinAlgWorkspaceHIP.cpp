#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>

#include <cassert>

namespace ReSolve
{
  LinAlgWorkspaceHIP::LinAlgWorkspaceHIP()
  {
    handle_rocsparse_ = nullptr;
    handle_rocblas_   = nullptr;
    mat_A_            = nullptr;
    info_A_           = nullptr;

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
    resetMatvecSetup();

    rocsparse_destroy_handle(handle_rocsparse_);
    rocblas_destroy_handle(handle_rocblas_);

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
    resetMatvecSetup();
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

  void LinAlgWorkspaceHIP::setSpmvMatrixDescriptor(rocsparse_mat_descr mat)
  {
    mat_A_ = mat;
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

  /**
   * @brief Reset the cached HIP SpMV setup.
   *
   * Destroys the cached rocSPARSE matrix descriptor and matrix info so the
   * next matvec call can rebuild the setup if the matrix or its dimensions have changed.
   */
  int LinAlgWorkspaceHIP::resetMatvecSetup()
  {
    if (mat_A_ != nullptr)
    {
      rocsparse_status status = rocsparse_destroy_mat_descr(mat_A_);
      mat_A_ = nullptr;
      if (status != rocsparse_status_success)
      {
        return -1;
      }
    }
    if (info_A_ != nullptr)
    {
      rocsparse_status status = rocsparse_destroy_mat_info(info_A_);
      info_A_ = nullptr;
      if (status != rocsparse_status_success)
      {
        return -1;
      }
    }
    matvec_setup_done_ = false;
  }

  int LinAlgWorkspaceHIP::initializeHandles()
  {
    if (rocsparse_create_handle(&handle_rocsparse_) != rocsparse_status_success) return -1;
    if (rocblas_create_handle(&handle_rocblas_)     != rocblas_status_success)   return -1;
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

  int LinAlgWorkspaceHIP::setTransposeBufferWorkspace(size_t bufferSize)
  {
    assert(!transpose_workspace_ready_ && "Transpose workspace already set!\n");
    transpose_workspace_ready_ = true;
    return mem_.allocateBufferOnDevice(&transpose_workspace_, bufferSize);
  }

  bool LinAlgWorkspaceHIP::isTransposeBufferAllocated()
  {
    return transpose_workspace_ready_;
  }
} // namespace ReSolve
