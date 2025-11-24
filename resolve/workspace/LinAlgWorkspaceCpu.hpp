#pragma once

#include <resolve/Common.hpp>

namespace ReSolve
{
  class ScaleAddBufferCpu;

  class LinAlgWorkspaceCpu
  {
  public:
    LinAlgWorkspaceCpu();
    ~LinAlgWorkspaceCpu();
    void               initializeHandles();
    void               resetLinAlgWorkspace();
    bool               scaleAddISetup();
    void               scaleAddISetupDone();
    ScaleAddBufferCpu* getScaleAddIBuffer();
    void               setScaleAddIBuffer(ScaleAddBufferCpu* buffer);
    bool               scaleAddBSetup();
    void               scaleAddBSetupDone();
    ScaleAddBufferCpu* getScaleAddBBuffer();
    void               setScaleAddBBuffer(ScaleAddBufferCpu* buffer);

  private:
    // check if setup is done for scaleAddI i.e. if buffer is allocated, csr structure is set etc.
    bool               scaleAddISetupDone_{false};
    ScaleAddBufferCpu* scaleAddIBuffer_{nullptr};
    // check if setup is done for scaleAddB i.e. if buffer is allocated, csr structure is set etc.
    bool               scaleAddBSetupDone_{false};
    ScaleAddBufferCpu* scaleAddBBuffer_{nullptr};
  };

} // namespace ReSolve
