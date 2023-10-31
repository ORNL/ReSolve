#pragma once

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

      void setRocblasHandle(rocblas_handle handle);
      void setRocsparseHandle(rocsparse_handle handle);
      void setSpmvMatrixDescriptor(rocsparse_mat_descr mat);
      void setSpmvMatrixInfo(rocsparse_mat_info info);

      void initializeHandles();

      bool matvecSetup();
      void matvecSetupDone();

    private:
      //handles
      rocblas_handle handle_rocblas_;
      rocsparse_handle handle_rocsparse_;

      //matrix descriptors
      rocsparse_mat_descr  mat_A_; 

      //vector descriptors not needed, rocsparse uses RAW pointers.

      //buffers
      // there is no buffer needed in matvec
      bool matvec_setup_done_; //check if setup is done for matvec (note: no buffer but there is analysis)

      //info - but we need info
      rocsparse_mat_info  info_A_;

      // MemoryHandler mem_; ///< Memory handler not needed for now
  };

} // namespace ReSolve
