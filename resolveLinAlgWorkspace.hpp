#include <cuda_runtime.h>
#include "cublas_v2.h"
namespace ReSolve
{
  class resolveLinAlgWorkspace
  {
    public:
      resolveLinAlgrWorkspace();
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

      cusparseHandle_t getCusparseHandle();
      cusolverSpHandle_t getCusolverSpHandle(); //needed for 1-norms etc
      cusparsHandle_t getCusparseHandle();      
      cusparseSpMatDescr_t getSpmvMatrixDescriptor();
      cusparseDnVecDescr_t getVecX();
      cusparseDnVecDescr_t  getVecY();

      void setCusparseHandle(cusparseHandle_t handle);
      void setCusolverSpHandle( cusolverSpHandle_t handle);
      void setCusparseHandle(cusparseHandle_t handle);
      void setSomvMatrixDescriptor(cusparseSpMatDescr_t mat);
      
      void initializeHandles();

      bool matvecSetup();
      void matvecSetupDone();
    
    private:
      //handles
      cusparseHandle_t handle_cusparse;
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
