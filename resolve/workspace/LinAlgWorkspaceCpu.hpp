#pragma once

#include <vector>

#include <resolve/Common.hpp>

namespace ReSolve
{
  struct ScaleAddIBuffer
  {
    std::vector<index_type> row_data_; ///< row data (HOST)
    std::vector<index_type> col_data_; ///< column data (HOST)
    index_type              nnz;
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
