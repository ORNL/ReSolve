#pragma once

#include <resolve/Common.hpp>

#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspaceHIP
  {
    public:
      LinAlgWorkspaceHIP();
      ~LinAlgWorkspaceHIP();

      rocblas_handle getRocblasHandle();
      rocsparse_handle getRocsparseHandle();
      rocsparse_mat_descr getSpmvMatrixDescriptor();
      rocsparse_mat_info getSpmvMatrixInfo();
      index_type getDrSize();
      real_type* getDr();
      real_type* getNormBuffer();
      void* getTransposeWorkspace();
      bool getNormBufferState();
      bool isTransposeAllocated();
      void setTransposeAllocated();

      void setRocblasHandle(rocblas_handle handle);
      void setRocsparseHandle(rocsparse_handle handle);
      void setSpmvMatrixDescriptor(rocsparse_mat_descr mat);
      void setSpmvMatrixInfo(rocsparse_mat_info info);

      void initializeHandles();

      bool matvecSetup();
      void matvecSetupDone();

      void setDrSize(index_type new_sz);
      void setDr(real_type* new_dr);
      void setNormBuffer(real_type* nb);
      void setNormBufferState(bool r);


    private:
      //handles
      rocblas_handle handle_rocblas_;
      rocsparse_handle handle_rocsparse_;

      //matrix descriptors
      rocsparse_mat_descr  mat_A_;

      //vector descriptors not needed, rocsparse uses RAW pointers.

      //buffers
      // there is no buffer needed in matvec
      bool matvec_setup_done_{false}; //check if setup is done for matvec (note: no buffer but there is analysis)

      //info - but we need info
      rocsparse_mat_info info_A_;

      real_type* d_r_{nullptr}; // needed for inf-norm
      real_type* norm_buffer_{nullptr}; // needed for inf-norm
      real_type* transpose_workspace_{nullptr}; // needed for transpose
      bool transpose_workspace_ready_{false}; // to track if allocated
      index_type d_r_size_{0};
      bool norm_buffer_ready_{false};// to track if allocated
      MemoryHandler mem_; ///< Memory handler not needed for now
  };

} // namespace ReSolve
