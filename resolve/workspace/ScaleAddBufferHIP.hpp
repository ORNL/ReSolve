#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{

  class ScaleAddBufferHIP
  {
  public:
    ScaleAddBufferHIP(index_type numRows);
    ~ScaleAddBufferHIP();
    index_type*         getRowData();
    rocsparse_mat_descr getMatrixDescriptor();
    index_type          getNumRows();
    void                setNnz(index_type nnz);
    index_type          getNnz();

  private:
    index_type*         rowData_;
    rocsparse_mat_descr mat_A_;
    index_type          numRows_;
    index_type          nnz_;
    MemoryHandler       mem_;
  };
} // namespace ReSolve
