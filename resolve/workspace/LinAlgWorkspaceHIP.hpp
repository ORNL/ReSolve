#pragma once

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspaceHIP
  {
  public:
    LinAlgWorkspaceHIP();
    ~LinAlgWorkspaceHIP();

    void resetLinAlgWorkspace();

    rocblas_handle      getRocblasHandle();
    rocsparse_handle    getRocsparseHandle();
    rocsparse_mat_descr getSpmvMatrixDescriptor();
    rocsparse_mat_info  getSpmvMatrixInfo();
    index_type          getDrSize();
    real_type*          getDr();
    real_type*          getNormBuffer();
    void*               getTransposeBufferWorkspace();
    int                 setTransposeBufferWorkspace(size_t bufferSize);
    bool                isTransposeBufferAllocated();
    void  setSpmvBuffer(void* buffer, index_type buffer_size);
    void  setNormBuffer(void* buffer);
    void  setQrBuffer(real_type* buffer, index_type buffer_size);

    void setRocblasHandle(rocblas_handle handle);
    void setRocsparseHandle(rocsparse_handle handle);
    void setSpmvMatrixDescriptor(rocsparse_mat_descr mat);
    void setSpmvMatrixInfo(rocsparse_mat_info info);
    
    bool                 getNormBufferState();
    bool                 getQrBufferState();
    bool                 isRngReady();
    hiprandState*         getRngState();
    index_type           getRngStateSize();
    index_type getSpmvBufferSize();

    void initializeHandles();
    void initializeRng(index_type size);
    void resetRng();
    int computeTotalThreads();
    index_type getTotalThreads();

    bool matvecSetup();
    void matvecSetupDone();
    /**
     * @brief Reset the cached HIP SpMV setup.
     *
     * Destroys the cached rocSPARSE matrix descriptor and matrix info so the
     * next matvec call can rebuild the setup if the matrix or its dimensions have changed.
     */
    void resetMatvecSetup();

    void setDrSize(index_type new_sz);
    void setDr(real_type* new_dr);
    void setNormBuffer(real_type* nb);
    void setNormBufferState(bool r);

  private:
    // handles
    rocblas_handle   handle_rocblas_;
    rocsparse_handle handle_rocsparse_;

    // matrix descriptors
    rocsparse_mat_descr mat_A_;

    // vector descriptors not needed, rocsparse uses RAW pointers.

    // buffers
    void* buffer_spmv_{nullptr};
    index_type spmv_buffer_size_{0};
    void* buffer_1norm_{nullptr};
    real_type* buffer_qr_{nullptr};

    //  there is no buffer needed in matvec
    bool matvec_setup_done_{false}; // check if setup is done for matvec (note: no buffer but there is analysis)

    // info - but we need info
    rocsparse_mat_info info_A_;

    real_type*    d_r_{nullptr};                     // needed for inf-norm
    real_type*    norm_buffer_{nullptr};             // needed for inf-norm
    void*         transpose_workspace_{nullptr};     // needed for transpose
    bool          transpose_workspace_ready_{false}; // to track if allocated
    index_type    d_r_size_{0};
    bool          norm_buffer_ready_{false}; // to track if allocated
    MemoryHandler mem_;                      ///< Memory handler not needed for now
    
    index_type qr_buffer_size_{0};
    bool       qr_buffer_ready_{false}; // to track if allocated
    int*       qr_dev_info_{nullptr};

    bool rng_ready_{false};
    index_type rng_state_size_{0};
    hiprandState* rng_state_;
    index_type total_threads_;

  };

} // namespace ReSolve
