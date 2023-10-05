#include <resolve/memoryUtils.hpp>
#include "LinAlgWorkspace.hpp"

namespace ReSolve
{
  LinAlgWorkspace::LinAlgWorkspace()
  {
  }
  
  LinAlgWorkspace::~LinAlgWorkspace()
  {
  }

  LinAlgWorkspaceCUDA::LinAlgWorkspaceCUDA()
  {
    handle_cusolversp_ = nullptr;
    handle_cusparse_   = nullptr;
    handle_cublas_     = nullptr;
    buffer_spmv_       = nullptr;
    buffer_1norm_      = nullptr;    

    matvec_setup_done_ = false;
  }

  LinAlgWorkspaceCUDA::~LinAlgWorkspaceCUDA()
  {
    if (buffer_spmv_ != nullptr)  deleteOnDevice(buffer_spmv_);
    if (buffer_1norm_ != nullptr) deleteOnDevice(buffer_1norm_);
    cusparseDestroy(handle_cusparse_);
    cusolverSpDestroy(handle_cusolversp_);
    cublasDestroy(handle_cublas_);
    cusparseDestroySpMat(mat_A_);
    // cusparseDestroyDnVec(vec_x_);
    // cusparseDestroyDnVec(vec_y_);
  }

  void* LinAlgWorkspaceCUDA::getSpmvBuffer()
  {
    return buffer_spmv_;
  }

  void* LinAlgWorkspaceCUDA::getNormBuffer()
  {
    return buffer_1norm_;
  }

  void LinAlgWorkspaceCUDA::setSpmvBuffer(void* buffer)
  {
    buffer_spmv_ =  buffer;
  }

  void LinAlgWorkspaceCUDA::setNormBuffer(void* buffer)
  {
    buffer_1norm_ =  buffer;
  }

  cusparseHandle_t LinAlgWorkspaceCUDA::getCusparseHandle()
  {
    return handle_cusparse_;
  }

  void LinAlgWorkspaceCUDA::setCusparseHandle(cusparseHandle_t handle)
  {
    handle_cusparse_ = handle;
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

  cusparseDnVecDescr_t   LinAlgWorkspaceCUDA::getVecX()
  {
    return vec_x_;
  }

  cusparseDnVecDescr_t   LinAlgWorkspaceCUDA::getVecY()
  {
    return vec_y_;
  }

  bool LinAlgWorkspaceCUDA::matvecSetup(){
    return matvec_setup_done_;
  }

  void LinAlgWorkspaceCUDA::matvecSetupDone() {
    matvec_setup_done_ = true;
  }

  void LinAlgWorkspaceCUDA::initializeHandles()
  {
    cusparseCreate(&handle_cusparse_);
    cublasCreate(&handle_cublas_);
    cusolverSpCreate(&handle_cusolversp_);
  }
}
