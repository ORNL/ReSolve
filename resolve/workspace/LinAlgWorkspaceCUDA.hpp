#pragma once

#include "cublas_v2.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{

  class ScaleAddBufferCUDA;

  class LinAlgWorkspaceCUDA
  {
  public:
    LinAlgWorkspaceCUDA();
    ~LinAlgWorkspaceCUDA();

    void resetLinAlgWorkspace();

    // accessors
    void*               getSpmvBuffer();
    void*               getNormBuffer();
    ScaleAddBufferCUDA* getScaleAddIBuffer();
    ScaleAddBufferCUDA* getScaleAddBBuffer();
    void*               getTransposeBufferWorkspace();
    void                setTransposeBufferWorkspace(size_t bufferSize);
    bool                isTransposeBufferAllocated();
    void                setSpmvBuffer(void* buffer);
    void                setNormBuffer(void* buffer);
    void                setScaleAddIBuffer(ScaleAddBufferCUDA* buffer);
    void                setScaleAddBBuffer(ScaleAddBufferCUDA* buffer);
    void                scaleAddISetupDone();
    void                scaleAddBSetupDone();

    cublasHandle_t       getCublasHandle();
    cusolverSpHandle_t   getCusolverSpHandle(); // needed for 1-norms etc
    cusparseHandle_t     getCusparseHandle();
    cusparseSpMatDescr_t getSpmvMatrixDescriptor();
    cusparseDnVecDescr_t getVecX();
    cusparseDnVecDescr_t getVecY();
    index_type           getDrSize();
    real_type*           getDr();
    bool                 getNormBufferState();

    void setCublasHandle(cublasHandle_t handle);
    void setCusolverSpHandle(cusolverSpHandle_t handle);
    void setCusparseHandle(cusparseHandle_t handle);
    void setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat);
    void setDrSize(index_type new_sz);
    void setDr(real_type* new_dr);
    void setNormBufferState(bool r);

    void initializeHandles();

    bool matvecSetup();
    void matvecSetupDone();

    bool scaleAddISetup();
    bool scaleAddBSetup();

  private:
    // handles
    cublasHandle_t     handle_cublas_;
    cusolverSpHandle_t handle_cusolversp_; // needed for 1-norm
    cusparseHandle_t   handle_cusparse_;

    // matrix descriptors
    cusparseSpMatDescr_t mat_A_;

    // vector descriptors
    cusparseDnVecDescr_t vec_x_;
    cusparseDnVecDescr_t vec_y_;

    // buffers
    void*               buffer_spmv_{nullptr};
    void*               buffer_1norm_{nullptr};
    ScaleAddBufferCUDA* buffer_scale_add_i{nullptr};
    ScaleAddBufferCUDA* buffer_scale_add_b{nullptr};

    bool matvec_setup_done_{false}; // check if setup is done for matvec i.e. if buffer is allocated, csr structure is set etc.
    bool scale_add_i_setup_done_{false};
    bool scale_add_b_setup_done_{false};

    void* transpose_workspace_{nullptr};     // needed for transpose
    bool  transpose_workspace_ready_{false}; // to track if allocated

    real_type* d_r_{nullptr}; // needed for one-norm
    index_type d_r_size_{0};
    bool       norm_buffer_ready_{false}; // to track if allocated

    MemoryHandler mem_;
  };

} // namespace ReSolve
