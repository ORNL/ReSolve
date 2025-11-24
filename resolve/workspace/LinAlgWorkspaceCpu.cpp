#include "LinAlgWorkspaceCpu.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>

#include <resolve/workspace/ScaleAddBufferCpu.hpp>

namespace ReSolve
{
  LinAlgWorkspaceCpu::LinAlgWorkspaceCpu()
  {
  }

  LinAlgWorkspaceCpu::~LinAlgWorkspaceCpu()
  {
    delete scaleAddIBuffer_;
    delete scaleAddBBuffer_;
  }

  void LinAlgWorkspaceCpu::initializeHandles()
  {
  }

  void LinAlgWorkspaceCpu::resetLinAlgWorkspace()
  {
    delete scaleAddIBuffer_;
    scaleAddIBuffer_    = nullptr;
    scaleAddISetupDone_ = false;
    delete scaleAddBBuffer_;
    scaleAddBBuffer_    = nullptr;
    scaleAddBSetupDone_ = false;
  }

  bool LinAlgWorkspaceCpu::scaleAddISetup()
  {
    return scaleAddISetupDone_;
  }

  void LinAlgWorkspaceCpu::scaleAddISetupDone()
  {
    scaleAddISetupDone_ = true;
  }

  ScaleAddBufferCpu* LinAlgWorkspaceCpu::getScaleAddIBuffer()
  {
    assert(scaleAddIBuffer_ != nullptr);
    return scaleAddIBuffer_;
  }

  void LinAlgWorkspaceCpu::setScaleAddIBuffer(ScaleAddBufferCpu* buffer)
  {
    assert(buffer != nullptr);
    assert(scaleAddIBuffer_ == nullptr);
    scaleAddIBuffer_ = buffer;
    scaleAddISetupDone();
  }

  bool LinAlgWorkspaceCpu::scaleAddBSetup()
  {
    return scaleAddBSetupDone_;
  }

  void LinAlgWorkspaceCpu::scaleAddBSetupDone()
  {
    scaleAddBSetupDone_ = true;
  }

  ScaleAddBufferCpu* LinAlgWorkspaceCpu::getScaleAddBBuffer()
  {
    assert(scaleAddBBuffer_ != nullptr);
    return scaleAddBBuffer_;
  }

  void LinAlgWorkspaceCpu::setScaleAddBBuffer(ScaleAddBufferCpu* buffer)
  {
    assert(scaleAddBBuffer_ == nullptr);
    scaleAddBBuffer_ = buffer;
    scaleAddBSetupDone();
  }

} // namespace ReSolve
