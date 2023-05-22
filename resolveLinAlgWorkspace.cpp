#include "resolveLinAlgWorkspace.hpp"

namespace ReSolve
{
  resolveLinAlgWorkspace::resolveLinAlgWorkspace()
  {
  }
  
  resolveLinAlgWorkspace::~resolveLinAlgWorkspace()
  {
  }

  resolveLinAlgWorkspaceCUDA::resolveLinAlgWorkspaceCUDA()
  {
    handle_cusolversp_ = nullptr;
    handle_cusparse_ = nullptr;
    handle_cublas_ = nullptr;
    buffer_spmv_ = nullptr;
    buffer_1norm_ = nullptr;    
    matvec_setup_done_ = false;
  }

  resolveLinAlgWorkspaceCUDA::~resolveLinAlgWorkspaceCUDA()
  {
    cudaFree(buffer_spmv_);
    cudaFree(buffer_1norm_);
    cusparseDestroy(handle_cusparse_);
    cusolverSpDestroy(handle_cusolversp_);
    cublasDestroy(handle_cublas_);
    cusparseDestroySpMat(mat_A_);
    cusparseDestroyDnVec(vec_x_);
    cusparseDestroyDnVec(vec_y_);
  }

  void* resolveLinAlgWorkspaceCUDA::getSpmvBuffer()
  {
    return buffer_spmv_;
  }

  void* resolveLinAlgWorkspaceCUDA::getNormBuffer()
  {
    return buffer_1norm_;
  }

  void resolveLinAlgWorkspaceCUDA::setSpmvBuffer(void* buffer)
  {
    buffer_spmv_ =  buffer;
  }

  void resolveLinAlgWorkspaceCUDA::setNormBuffer(void* buffer)
  {
    buffer_1norm_ =  buffer;
  }

  cusparseHandle_t resolveLinAlgWorkspaceCUDA::getCusparseHandle()
  {
    return handle_cusparse_;
  }

  void resolveLinAlgWorkspaceCUDA::setCusparseHandle(cusparseHandle_t handle)
  {
    handle_cusparse_ = handle;
  }

  cublasHandle_t resolveLinAlgWorkspaceCUDA::getCublasHandle()
  {
    return handle_cublas_;
  }

  void resolveLinAlgWorkspaceCUDA::setCublasHandle(cublasHandle_t handle)
  {
    handle_cublas_ = handle;
  }

  cusolverSpHandle_t resolveLinAlgWorkspaceCUDA::getCusolverSpHandle()
  {
    return handle_cusolversp_;
  }

  void resolveLinAlgWorkspaceCUDA::setCusolverSpHandle(cusolverSpHandle_t handle)
  {
    handle_cusolversp_ = handle;
  }

  cusparseSpMatDescr_t resolveLinAlgWorkspaceCUDA::getSpmvMatrixDescriptor()
  {
    return mat_A_;
  }

  void resolveLinAlgWorkspaceCUDA::setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat)
  {
    mat_A_ = mat;
  }

  cusparseDnVecDescr_t   resolveLinAlgWorkspaceCUDA::getVecX()
  {
    return vec_x_;
  }

  cusparseDnVecDescr_t   resolveLinAlgWorkspaceCUDA::getVecY()
  {
    return vec_y_;
  }

  bool resolveLinAlgWorkspaceCUDA::matvecSetup(){
    return matvec_setup_done_;
  }

  void resolveLinAlgWorkspaceCUDA::matvecSetupDone() {
    matvec_setup_done_ = true;
  }

  void resolveLinAlgWorkspaceCUDA::initializeHandles()
  {
    cusparseCreate(&handle_cusparse_);
    cublasCreate(&handle_cublas_);
    cusolverSpCreate(&handle_cusolversp_);
  }
}
