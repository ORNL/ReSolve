#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverSp.h"
#pragma once
namespace ReSolve
{
  class resolveLinAlgWorkspace
  {
    public:
      resolveLinAlgWorkspace();
      ~resolveLinAlgWorkspace();
    private:
  };


  class resolveLinAlgWorkspaceCUDA : public resolveLinAlgWorkspace
  {
    public:
resolveLinAlgWorkspaceCUDA();
      ~resolveLinAlgWorkspaceCUDA();

      //accessors

      //accessors:
      void* getSpmvBuffer();
      void* getNormBuffer();

      void setSpmvBuffer(void* buffer);
      void setNormBuffer(void* buffer);

      cublasHandle_t getCublasHandle();
      cusolverSpHandle_t getCusolverSpHandle(); //needed for 1-norms etc
      cusparseHandle_t getCusparseHandle();      
      cusparseSpMatDescr_t getSpmvMatrixDescriptor();
      cusparseDnVecDescr_t getVecX();
      cusparseDnVecDescr_t  getVecY();

      void setCublasHandle(cublasHandle_t handle);
      void setCusolverSpHandle( cusolverSpHandle_t handle);
      void setCusparseHandle(cusparseHandle_t handle);
      void setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat);
      
      void initializeHandles();

      bool matvecSetup();
      void matvecSetupDone();
    
    private:
      //handles
      cublasHandle_t handle_cublas;
      cusolverSpHandle_t handle_cusolversp;//needed for 1-norm
      cusparseHandle_t handle_cusparse;

      //matrix descriptors
      cusparseSpMatDescr_t mat_A; 

      //vector descriptors

       cusparseDnVecDescr_t vec_x, vec_y;

      //buffers
      void* buffer_spmv;
      void* buffer_1norm;

      bool matvec_setup_done; //check if setup is done for matvec i.e. if buffer is allocated, csr structure is set etc.
  };
}
