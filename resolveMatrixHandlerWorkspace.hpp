#include "resolveCommon.hpp"

#include "cusparse.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"
namespace ReSolve
{
  class resolveMatrixHandlerWorkspace
  {
    public:
      resolveMatrixHandlerWorkspace();
      ~resolveMatrixHandlerWorkspace();
    private:
  };

  
  class resolveMatrixHandlerWorkspaceCUDA : public resolveMatrixHandlerWorkspace
  {
    public:
      resolveMatrixHandlerWorkspaceCUDA();
      ~resolveMatrixHandlerWorkspaceCUDA();

    //accessors:
    void* getSpmvBuffer();
    void* getNormBuffer();
    
    void setSpmvBuffer(void* buffer);
    void setNormBuffer(void* buffer);

    cusparseHandle_t getCusparseHandle();
    cublasHandle_t getCublasHandle();

    void setCusparseHandle(cusparseHandle_t handle);
    void setCublasHandle(cublasHandle_t handle);

    void initializeHandles();

    private:
      //handles
       cusparseHandle_t handle_cusparse;
       cublasHandle_t handle_cublas;

      //buffers
       void* buffer_spmv;
       void* buffer_1norm;
  };
}
