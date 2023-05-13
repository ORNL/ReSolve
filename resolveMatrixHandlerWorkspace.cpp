#include "resolveMatrixHandlerWorkspace.hpp"

namespace ReSolve
{
  resolveMatrixHandlerWorkspace::resolveMatrixHandlerWorkspace(){};
  resolveMatrixHandlerWorkspace::~resolveMatrixHandlerWorkspace(){}



  resolveMatrixHandlerWorkspaceCUDA::resolveMatrixHandlerWorkspaceCUDA()
  {
    handle_cusparse = nullptr;
    handle_cublas = nullptr;
    buffer_spmv = nullptr;
    buffer_1norm = nullptr;
  }

  resolveMatrixHandlerWorkspaceCUDA::~resolveMatrixHandlerWorkspaceCUDA()
  {
    cudaFree(buffer_spmv);
    cudaFree(buffer_1norm);
    cusparseDestroy(handle_cusparse);
    cublasDestroy(handle_cublas);
  }

  void* resolveMatrixHandlerWorkspaceCUDA::getSpmvBuffer()
  {
    return buffer_spmv;
  }

  void* resolveMatrixHandlerWorkspaceCUDA::getNormBuffer()
  {
    return buffer_1norm;
  }

  void resolveMatrixHandlerWorkspaceCUDA::setSpmvBuffer(void* buffer)
  {
    buffer_spmv =  buffer;
  }

  void resolveMatrixHandlerWorkspaceCUDA::setNormBuffer(void* buffer)
  {
    buffer_1norm =  buffer;
  }

  cusparseHandle_t resolveMatrixHandlerWorkspaceCUDA::getCusparseHandle()
  {
    return handle_cusparse;
  }

  cublasHandle_t resolveMatrixHandlerWorkspaceCUDA::getCublasHandle()
  {
    return handle_cublas;
  }

  void resolveMatrixHandlerWorkspaceCUDA::setCusparseHandle(cusparseHandle_t handle)
  {
    handle_cusparse = handle;
  }

  void resolveMatrixHandlerWorkspaceCUDA::setCublasHandle(cublasHandle_t handle)
  {
    handle_cublas = handle;
  }

  void resolveMatrixHandlerWorkspaceCUDA::initializeHandles()
  {
    cusparseCreate(&handle_cusparse);
    cublasCreate(&handle_cublas); 
  }
}
