#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>

namespace ReSolve
{
  LinAlgWorkspaceCUDA::LinAlgWorkspaceCUDA()
  {
    handle_cusolversp_         = nullptr;
    handle_cusparse_           = nullptr;
    handle_cublas_             = nullptr;
    buffer_spmv_               = nullptr;
    buffer_1norm_              = nullptr;
    transpose_workspace_       = nullptr;
    transpose_workspace_ready_ = false;
    d_r_                       = nullptr;
    d_r_size_                  = 0;
    matvec_setup_done_         = false;
    norm_buffer_ready_         = false;
  }

  LinAlgWorkspaceCUDA::~LinAlgWorkspaceCUDA()
  {
    if (buffer_spmv_ != nullptr)
      mem_.deleteOnDevice(buffer_spmv_);
    if (d_r_size_ != 0)
      mem_.deleteOnDevice(d_r_);
    if (norm_buffer_ready_)
      mem_.deleteOnDevice(buffer_1norm_);
    cusparseDestroy(handle_cusparse_);
    cusolverSpDestroy(handle_cusolversp_);
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

  void* LinAlgWorkspaceCUDA::getSpmvBuffer()
  {
    return buffer_spmv_;
  }

  void* LinAlgWorkspaceCUDA::getNormBuffer()
  {
    return buffer_1norm_;
  }

  void* LinAlgWorkspaceCUDA::getTransposeBufferWorkspace()
  {
    return transpose_workspace_;
  }

  void LinAlgWorkspaceCUDA::setTransposeBufferWorkspace(size_t bufferSize)
  {
    if (transpose_workspace_ready_)
    {
      mem_.deleteOnDevice(transpose_workspace_);
    }
    mem_.allocateBufferOnDevice(&transpose_workspace_, bufferSize);
    transpose_workspace_ready_ = true;
    return;
  }

  bool LinAlgWorkspaceCUDA::isTransposeBufferAllocated()
  {
    return transpose_workspace_ready_;
  }

  bool LinAlgWorkspaceCUDA::getNormBufferState()
  {
    return norm_buffer_ready_;
  }

  void LinAlgWorkspaceCUDA::setSpmvBuffer(void* buffer)
  {
    buffer_spmv_ = buffer;
  }

  void LinAlgWorkspaceCUDA::setNormBuffer(void* buffer)
  {
    buffer_1norm_ = buffer;
  }

  void LinAlgWorkspaceCUDA::setNormBufferState(bool r)
  {
    norm_buffer_ready_ = r;
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

  void LinAlgWorkspaceCUDA::setCusolverSpHandle(cusolverSpHandle_t handle)
  {
    handle_cusolversp_ = handle;
  }

  cusparseSpMatDescr_t LinAlgWorkspaceCUDA::getSpmvMatrixDescriptor()
  {
    return mat_A_;
  }

  void LinAlgWorkspaceCUDA::setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat)
  {
    mat_A_ = mat;
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

  void LinAlgWorkspaceCUDA::initializeHandles()
  {
    cusparseCreate(&handle_cusparse_);
    cublasCreate(&handle_cublas_);
    cusolverSpCreate(&handle_cusolversp_);
  }
} // namespace ReSolve
