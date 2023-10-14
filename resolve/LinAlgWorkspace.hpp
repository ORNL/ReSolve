#pragma once

#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverSp.h"

#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspace
  {
    public:
      LinAlgWorkspace();
      ~LinAlgWorkspace();
    protected:
      MemoryHandler mem_;
  };


  class LinAlgWorkspaceCUDA : public LinAlgWorkspace
  {
    public:
      LinAlgWorkspaceCUDA();
      ~LinAlgWorkspaceCUDA();

      //accessors
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
      cublasHandle_t handle_cublas_;
      cusolverSpHandle_t handle_cusolversp_;//needed for 1-norm
      cusparseHandle_t handle_cusparse_;

      //matrix descriptors
      cusparseSpMatDescr_t mat_A_; 

      //vector descriptors
      cusparseDnVecDescr_t vec_x_, vec_y_;

      //buffers
      void* buffer_spmv_;
      void* buffer_1norm_;

      bool matvec_setup_done_; //check if setup is done for matvec i.e. if buffer is allocated, csr structure is set etc.
  };

  /// @brief  Workspace factory
  /// @param[in] memspace memory space ID 
  /// @return pointer to the linear algebra workspace
  inline LinAlgWorkspace* createLinAlgWorkspace(std::string memspace)
  {
    if (memspace == "cuda") {
      LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
      workspace->initializeHandles();
      return workspace;
    } 
    // If not CUDA, return default
    return (new LinAlgWorkspace());
  }

}
