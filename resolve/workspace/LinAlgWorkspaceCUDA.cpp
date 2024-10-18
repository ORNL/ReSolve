#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>

namespace ReSolve
{
  LinAlgWorkspaceCUDA::LinAlgWorkspaceCUDA()
  {
  }

  LinAlgWorkspaceCUDA::~LinAlgWorkspaceCUDA()
  {
    // Delete handles
    cublasDestroy(handle_cublas_);
    cusparseDestroy(handle_cusparse_);
    cusolverSpDestroy(handle_cusolversp_);

    // If for some reason mat_A_ is not deleted ...
    // TODO: probably should print warning if true
    if (matvec_setup_done_) {
      cusparseDestroySpMat(mat_A_);
    }

    if (vec_x_) {
      cusparseDestroyDnVec(vec_x_);
    }
    if (vec_y_) {
      cusparseDestroyDnVec(vec_y_);
    }

    // Delete buffers
    if (buffer_spmv_ != nullptr) {
      mem_.deleteOnDevice(buffer_spmv_);
    }
    if (d_r_size_ != 0) {
      mem_.deleteOnDevice(d_r_);
    }
    if (norm_buffer_ready_) {
      mem_.deleteOnDevice(buffer_1norm_);
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
  
  bool  LinAlgWorkspaceCUDA::getNormBufferState()
  {
    return norm_buffer_ready_;
  }

  void LinAlgWorkspaceCUDA::setSpmvBuffer(void* buffer)
  {
    buffer_spmv_ =  buffer;
  }

  void LinAlgWorkspaceCUDA::setNormBuffer(void* buffer)
  {
    buffer_1norm_ =  buffer;
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
  
  void LinAlgWorkspaceCUDA::setDr(real_type* new_dr)
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

  index_type  LinAlgWorkspaceCUDA::getDrSize()
  {
    return d_r_size_;
  }

  real_type*  LinAlgWorkspaceCUDA::getDr()
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
