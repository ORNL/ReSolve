#include "resolveLinAlgWorkspace.hpp"

namespace ReSolve
{
  resolveLinAlgWorkspace::resolveLinAlgWorkspace(){};
  resolveLinAlgWorkspace::~resolveLinAlgWorkspace(){}



  resolveLinAlgWorkspaceCUDA::resolveLinAlgWorkspaceCUDA()
  {
    handle_cusparse = nullptr;
    handle_cusolversp = nullptr;
    handle_cusparse = nullptr;
    buffer_spmv = nullptr;
    buffer_1norm = nullptr;    
    matvec_setup_done = false;
  }

  resolveLinAlgWorkspaceCUDA::~resolveLinAlgWorkspaceCUDA()
  {
    cudaFree(buffer_spmv);
    cudaFree(buffer_1norm);
    cusparseDestroy(handle_cusparse);
    cusolverSpDestroy(handle_cusolversp);
    cublasDestroy(handle_cublas);
    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
  }

  void* resolveLinAlgWorkspaceCUDA::getSpmvBuffer()
  {
    return buffer_spmv;
  }

  void* resolveLinAlgWorkspaceCUDA::getNormBuffer()
  {
    return buffer_1norm;
  }

  void resolveLinAlgWorkspaceCUDA::setSpmvBuffer(void* buffer)
  {
    buffer_spmv =  buffer;
  }

  void resolveLinAlgWorkspaceCUDA::setNormBuffer(void* buffer)
  {
    buffer_1norm =  buffer;
  }

  cusparseHandle_t resolveLinAlgWorkspaceCUDA::getCusparseHandle()
  {
    return handle_cusparse;
  }

  void resolveLinAlgWorkspaceCUDA::setCusparseHandle(cusparseHandle_t handle)
  {
    handle_cusparse = handle;
  }

  cublasHandle_t resolveLinAlgWorkspaceCUDA::getCublasHandle()
  {
    return handle_cublas;
  }

  void resolveLinAlgWorkspaceCUDA::setCublasHandle(cublasHandle_t handle)
  {
    handle_cublas = handle;
  }

  cusolverSpHandle_t resolveLinAlgWorkspaceCUDA::getCusolverSpHandle()
  {
    return handle_cusolversp;
  }

  void resolveLinAlgWorkspaceCUDA::setCusolverSpHandle(cusolverSpHandle_t handle)
  {
    handle_cusolversp = handle;
  }



  cusparseSpMatDescr_t resolveLinAlgWorkspaceCUDA::getSpmvMatrixDescriptor()
  {
    return mat_A;
  }

  void resolveLinAlgWorkspaceCUDA::setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat)
  {
    mat_A = mat;
  }

  cusparseDnVecDescr_t   resolveLinAlgWorkspaceCUDA::getVecX()
  {
    return vec_x;
  }

  cusparseDnVecDescr_t   resolveLinAlgWorkspaceCUDA::getVecY()
  {
    return vec_y;
  }

  bool resolveLinAlgWorkspaceCUDA::matvecSetup(){
    return matvec_setup_done;
  }

  void resolveLinAlgWorkspaceCUDA::matvecSetupDone() {
    matvec_setup_done = true;
  }

  void resolveLinAlgWorkspaceCUDA::initializeHandles()
  {
    cusparseCreate(&handle_cusparse);
    cublasCreate(&handle_cublas);
    cusolverSpCreate(&handle_cusolversp);
  }
}
