#include <cassert>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>

namespace ReSolve
{
  using out = io::Logger;

  LinAlgWorkspaceCUDA::LinAlgWorkspaceCUDA()
  {
    handle_cusolversp_         = nullptr;
    handle_cusolverdn_         = nullptr;
    handle_cusparse_           = nullptr;
    handle_cublas_             = nullptr;
    buffer_spmv_               = nullptr;
    buffer_1norm_              = nullptr;
    buffer_qr_                 = nullptr;
    transpose_workspace_       = nullptr;
    transpose_workspace_ready_ = false;
    d_r_                       = nullptr;
    d_r_size_                  = 0;
    matvec_setup_done_         = false;
    norm_buffer_ready_         = false;
    qr_buffer_ready_           = false;
    qr_buffer_size_            = 0;
  }

  LinAlgWorkspaceCUDA::~LinAlgWorkspaceCUDA()
  {
    if (buffer_spmv_ != nullptr)
      mem_.deleteOnDevice(buffer_spmv_);
    if (d_r_size_ != 0)
      mem_.deleteOnDevice(d_r_);
    if (norm_buffer_ready_)
      mem_.deleteOnDevice(buffer_1norm_);
    if (qr_buffer_ready_)
      mem_.deleteOnDevice(buffer_qr_);
      mem_.deleteOnDevice(qr_dev_info_);
    cusparseDestroy(handle_cusparse_);
    cusolverSpDestroy(handle_cusolversp_);
    cusolverDnDestroy(handle_cusolverdn_);
    cublasDestroy(handle_cublas_);
    if (matvec_setup_done_)
    {
      cusparseDestroySpMat(mat_A_);
    }
    if (transpose_workspace_ready_)
    {
      mem_.deleteOnDevice(transpose_workspace_);
    }
  }

  /**
   * @brief Resets the CUDA linear algebra workspace.
   *
   * This function clears the state of the linear algebra workspace by
   * destroying the matrix descriptor, deallocating the residual vector,
   * deleting the norm buffer, and resetting the transpose workspace.
   */
  void LinAlgWorkspaceCUDA::resetLinAlgWorkspace()
  {
    if (matvec_setup_done_)
    {
      cusparseDestroySpMat(mat_A_);
      matvec_setup_done_ = false;
    }
    if (d_r_size_ != 0)
    {
      mem_.deleteOnDevice(d_r_);
      d_r_      = nullptr;
      d_r_size_ = 0;
    }
    if (norm_buffer_ready_)
    {
      mem_.deleteOnDevice(buffer_1norm_);
      buffer_1norm_      = nullptr;
      norm_buffer_ready_ = false;
    }
    if (qr_buffer_ready_)
    {
      mem_.deleteOnDevice(buffer_qr_);
      buffer_qr_      = nullptr;
      qr_buffer_size_ = 0;
      qr_buffer_ready_ = false;
      cudaFree(qr_dev_info_);
    }
    if (transpose_workspace_ready_)
    {
      mem_.deleteOnDevice(transpose_workspace_);
      transpose_workspace_       = nullptr;
      transpose_workspace_ready_ = false;
    }
    return;
  }

  void* LinAlgWorkspaceCUDA::getSpmvBuffer()
  {
    return buffer_spmv_;
  }

  void* LinAlgWorkspaceCUDA::getNormBuffer()
  {
    return buffer_1norm_;
  }

  real_type* LinAlgWorkspaceCUDA::getQrBuffer()
  {
    return buffer_qr_;
  }

  index_type LinAlgWorkspaceCUDA::getQrBufferSize()
  {
    return qr_buffer_size_;
  }

  void* LinAlgWorkspaceCUDA::getTransposeBufferWorkspace()
  {
    return transpose_workspace_;
  }

  int LinAlgWorkspaceCUDA::setTransposeBufferWorkspace(size_t buffer_size)
  {
    if (transpose_workspace_ready_)
    {
      out::error() << "Transpose workspace already set!\n";
      return 1;
    }
    mem_.allocateBufferOnDevice(&transpose_workspace_, buffer_size);
    transpose_workspace_ready_ = true;
    return 0;
  }

  bool LinAlgWorkspaceCUDA::isTransposeBufferAllocated()
  {
    return transpose_workspace_ready_;
  }

  bool LinAlgWorkspaceCUDA::getNormBufferState()
  {
    return norm_buffer_ready_;
  }

  bool LinAlgWorkspaceCUDA::getQrBufferState()
  {
    return qr_buffer_ready_;
  }

  void LinAlgWorkspaceCUDA::setSpmvBuffer(void* buffer)
  {
    buffer_spmv_ = buffer;
  }

  void LinAlgWorkspaceCUDA::setNormBuffer(void* buffer)
  {
    buffer_1norm_ = buffer;
  }

  void LinAlgWorkspaceCUDA::setQrBuffer(real_type* buffer, index_type buffer_size)
  {
    buffer_qr_ = buffer;
    qr_buffer_size_ = buffer_size;
  }

  void LinAlgWorkspaceCUDA::setNormBufferState(bool r)
  {
    norm_buffer_ready_ = r;
  }

  void LinAlgWorkspaceCUDA::setQrBufferState(bool r)
  {
    qr_buffer_ready_ = r;
  }

  cusparseHandle_t LinAlgWorkspaceCUDA::getCusparseHandle()
  {
    return handle_cusparse_;
  }

  void LinAlgWorkspaceCUDA::setCusparseHandle(cusparseHandle_t handle)
  {
    handle_cusparse_ = handle;
  }

  void LinAlgWorkspaceCUDA::setDrSize(index_type new_sz)
  {
    d_r_size_ = new_sz;
  }

  void LinAlgWorkspaceCUDA::setDr(double* new_dr)
  {
    d_r_ = new_dr;
  }

  cublasHandle_t LinAlgWorkspaceCUDA::getCublasHandle()
  {
    return handle_cublas_;
  }

  void LinAlgWorkspaceCUDA::setCublasHandle(cublasHandle_t handle)
  {
    handle_cublas_ = handle;
  }

  cusolverSpHandle_t LinAlgWorkspaceCUDA::getCusolverSpHandle()
  {
    return handle_cusolversp_;
  }

  cusolverDnHandle_t LinAlgWorkspaceCUDA::getCusolverDnHandle()
  {
    return handle_cusolverdn_;
  }

  void LinAlgWorkspaceCUDA::setCusolverSpHandle(cusolverSpHandle_t handle)
  {
    handle_cusolversp_ = handle;
  }

  void LinAlgWorkspaceCUDA::setCusolverDnHandle(cusolverDnHandle_t handle)
  {
    handle_cusolverdn_ = handle;
  }

  cusparseSpMatDescr_t LinAlgWorkspaceCUDA::getSpmvMatrixDescriptor()
  {
    return mat_A_;
  }

  void LinAlgWorkspaceCUDA::setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat)
  {
    mat_A_ = mat;
  }

  cusparseDnMatDescr_t LinAlgWorkspaceCUDA::getMatX()
  {
    return mat_X_;
  }

  cusparseDnMatDescr_t LinAlgWorkspaceCUDA::getMatY()
  {
    return mat_Y_;
  }

  cusparseDnVecDescr_t LinAlgWorkspaceCUDA::getVecX()
  {
    return vec_x_;
  }

  cusparseDnVecDescr_t LinAlgWorkspaceCUDA::getVecY()
  {
    return vec_y_;
  }

  index_type LinAlgWorkspaceCUDA::getDrSize()
  {
    return d_r_size_;
  }

  real_type* LinAlgWorkspaceCUDA::getDr()
  {
    return d_r_;
  }

  bool LinAlgWorkspaceCUDA::matvecSetup()
  {
    return matvec_setup_done_;
  }

  void LinAlgWorkspaceCUDA::matvecSetupDone()
  {
    matvec_setup_done_ = true;
  }

  /**
   * @brief Reset cached SpMV resources.
   */
  void LinAlgWorkspaceCUDA::resetMatvecSetup()
  {
    if (matvec_setup_done_)
    {
      cusparseDestroySpMat(mat_A_);
      matvec_setup_done_ = false;
    }
    if (buffer_spmv_ != nullptr)
    {
      mem_.deleteOnDevice(buffer_spmv_);
      buffer_spmv_ = nullptr;
    }
  }

  void LinAlgWorkspaceCUDA::allocateQrDevInfo()
  {
    // todo: use memory handler
    mem_.allocateArrayOnDevice(&qr_dev_info_, 1);
  }

  int* LinAlgWorkspaceCUDA::getQrDevInfo()
  {
    return qr_dev_info_;
  }

  void LinAlgWorkspaceCUDA::initializeHandles()
  {
    cusparseCreate(&handle_cusparse_);
    cublasCreate(&handle_cublas_);
    cusolverSpCreate(&handle_cusolversp_);
    cusolverDnCreate(&handle_cusolverdn_);
  }
} // namespace ReSolve
