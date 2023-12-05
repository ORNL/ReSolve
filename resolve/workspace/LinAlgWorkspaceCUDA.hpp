#pragma once

#include <resolve/Common.hpp>

#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverSp.h"

#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspaceCUDA
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
      index_type getDrSize();
      real_type* getDr();


      void setCublasHandle(cublasHandle_t handle);
      void setCusolverSpHandle( cusolverSpHandle_t handle);
      void setCusparseHandle(cusparseHandle_t handle);
      void setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat);
      void setDrSize(index_type new_sz);
      void setDr(real_type* new_dr);

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
      void* buffer_1norm_{nullptr};

      bool matvec_setup_done_; //check if setup is done for matvec i.e. if buffer is allocated, csr structure is set etc.
      
      real_type* d_r_; // needed for one-norm
      index_type d_r_size_;

      MemoryHandler mem_;
  };

} // namespace ReSolve
