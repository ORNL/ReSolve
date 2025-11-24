#pragma once

#include "cusparse.h"
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{

  class ScaleAddBufferCUDA
  {
  public:
    ScaleAddBufferCUDA(index_type numRows);
    ~ScaleAddBufferCUDA();
    index_type*        getRowData();
    void               allocateBuffer(size_t bufferSize);
    void*              getBuffer();
    cusparseMatDescr_t getMatrixDescriptor();
    index_type         getNumRows();
    void               setNnz(index_type nnz);
    index_type         getNnz();

  private:
    index_type*        rowData_;
    void*              buffer_;
    cusparseMatDescr_t mat_A_;
    index_type         numRows_;
    index_type         nnz_;
    size_t             bufferSize_;
    MemoryHandler      mem_;
  };

} // namespace ReSolve