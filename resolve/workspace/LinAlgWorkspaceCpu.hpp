#pragma once

#include <vector>

#include <resolve/Common.hpp>

namespace ReSolve
{
  class ScaleAddIBuffer
  {
  public:
    ScaleAddIBuffer(std::vector<index_type> row_data, std::vector<index_type> col_data);
    index_type* getRowData();
    index_type* getColumnData();
    index_type getNumRows();
    index_type getNumColumns();
    index_type getNnz();

  private:
    std::vector<index_type> row_data_;
    std::vector<index_type> col_data_;
  };

  class LinAlgWorkspaceCpu
  {
  public:
    LinAlgWorkspaceCpu();
    ~LinAlgWorkspaceCpu();
    void initializeHandles();
    void resetLinAlgWorkspace();
    bool             scaleAddISetup();
    void             scaleAddISetupDone();
    ScaleAddIBuffer* getScaleAddIBuffer();
    void             setScaleAddIBuffer(ScaleAddIBuffer* buffer);

  private:
    // check if setup is done for scaleAddI i.e. if buffer is allocated, csr structure is set etc.
    bool             scaleaddi_setup_done_{false};
    ScaleAddIBuffer* scaleaddi_buffer_{nullptr};
  };

} // namespace ReSolve
