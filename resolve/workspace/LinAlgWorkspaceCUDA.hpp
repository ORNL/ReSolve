#pragma once

#include "cublas_v2.h"
#include "cusolverSp.h"
#include "cusolverDn.h"
#include "cusparse.h"
#include "curand_kernel.h"
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspaceCUDA
  {
  public:
    LinAlgWorkspaceCUDA();
    ~LinAlgWorkspaceCUDA();

    void resetLinAlgWorkspace();

    // accessors
    void* getSpmvBuffer();
    void* getNormBuffer();
    real_type* getQrBuffer();
    index_type getQrBufferSize();
    void* getTransposeBufferWorkspace();
    int   setTransposeBufferWorkspace(size_t buffer_size);
    bool  isTransposeBufferAllocated();
    void  setSpmvBuffer(void* buffer);
    void  setNormBuffer(void* buffer);
    void  setQrBuffer(real_type* buffer, index_type buffer_size);

    cublasHandle_t       getCublasHandle();
    cusolverSpHandle_t   getCusolverSpHandle(); // needed for 1-norms etc
    cusolverDnHandle_t   getCusolverDnHandle(); // needed for choleskyQr
    cusparseHandle_t     getCusparseHandle();
    cusparseSpMatDescr_t getSpmvMatrixDescriptor();
    cusparseDnMatDescr_t getMatX();
    cusparseDnMatDescr_t getMatY();
    cusparseDnVecDescr_t getVecX();
    cusparseDnVecDescr_t getVecY();
    index_type           getDrSize();
    real_type*           getDr();
    bool                 getNormBufferState();
    bool                 getQrBufferState();
    bool                 isRngReady();
    curandState*         getRngState();
    index_type           getRngStateSize();

    void setCublasHandle(cublasHandle_t handle);
    void setCusolverSpHandle(cusolverSpHandle_t handle);
    void setCusolverDnHandle(cusolverDnHandle_t handle);
    void setCusparseHandle(cusparseHandle_t handle);
    void setSpmvMatrixDescriptor(cusparseSpMatDescr_t mat);
    void setDrSize(index_type new_sz);
    void setDr(real_type* new_dr);
    void setNormBufferState(bool r);
    void setQrBufferState(bool r);

    void initializeHandles();
    void initializeRng(index_type size);
    void resetRng();
    void computeTotalThreads();
    index_type getTotalThreads();

    bool matvecSetup();
    void matvecSetupDone();
    /**
     * @brief Reset the cached CUDA SpMV setup.
     *
     * Destroys the cached sparse matrix descriptor and frees the SpMV buffer so
     * the next matvec call can rebuild the SpMV setup if the matrix or its dimensions have changed.
     */
    void resetMatvecSetup();

    void allocateQrDevInfo();
    int* getQrDevInfo();

  private:
    // handles
    cublasHandle_t     handle_cublas_;
    cusolverSpHandle_t handle_cusolversp_; // needed for 1-norm
    cusolverDnHandle_t handle_cusolverdn_; // needed for choleskyQr
    cusparseHandle_t   handle_cusparse_;

    // matrix descriptors
    cusparseSpMatDescr_t mat_A_;
    cusparseDnMatDescr_t mat_X_;
    cusparseDnMatDescr_t mat_Y_;

    // vector descriptors
    cusparseDnVecDescr_t vec_x_;
    cusparseDnVecDescr_t vec_y_;

    // buffers
    void* buffer_spmv_{nullptr};
    void* buffer_1norm_{nullptr};
    real_type* buffer_qr_{nullptr};

    bool matvec_setup_done_{false}; // check if setup is done for matvec i.e. if buffer is allocated, csr structure is set etc.

    void* transpose_workspace_{nullptr};     // needed for transpose
    bool  transpose_workspace_ready_{false}; // to track if allocated

    real_type* d_r_{nullptr}; // needed for one-norm
    index_type d_r_size_{0};
    bool       norm_buffer_ready_{false}; // to track if allocated
    index_type qr_buffer_size_{0};
    bool       qr_buffer_ready_{false}; // to track if allocated
    int*       qr_dev_info_{nullptr};

    bool rng_ready_{false};
    index_type rng_state_size_{0};
    curandState* rng_state_;
    index_type total_threads_;

    MemoryHandler mem_;
  };

} // namespace ReSolve
