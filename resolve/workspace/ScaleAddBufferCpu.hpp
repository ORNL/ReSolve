#pragma once

#include <vector>

#include <resolve/Common.hpp>

namespace ReSolve
{
  class ScaleAddBufferCpu
  {
  public:
    ScaleAddBufferCpu(std::vector<index_type> rowData, std::vector<index_type> colData);
    index_type* getRowData();
    index_type* getColumnData();
    index_type  getNumRows();
    index_type  getNumColumns();
    index_type  getNnz();

  private:
    std::vector<index_type> rowData_;
    std::vector<index_type> colData_;
  };
} // namespace ReSolve
