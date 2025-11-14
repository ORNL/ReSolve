#include "LinAlgWorkspaceCpu.hpp"

#include <cassert>
#include <cstddef>

namespace ReSolve
{
  LinAlgWorkspaceCpu::LinAlgWorkspaceCpu()
  {
  }

  LinAlgWorkspaceCpu::~LinAlgWorkspaceCpu()
  {
    delete scaleaddi_buffer_;
  }

  void LinAlgWorkspaceCpu::initializeHandles()
  {
  }

  void LinAlgWorkspaceCpu::resetLinAlgWorkspace()
  {
    delete scaleaddi_buffer_;
    scaleaddi_buffer_ = nullptr;
  }

  bool LinAlgWorkspaceCpu::scaleAddISetup()
  {
    return scaleaddi_setup_done_;
  }

  void LinAlgWorkspaceCpu::scaleAddISetupDone()
  {
    scaleaddi_setup_done_ = true;
  }

  ScaleAddIBuffer* LinAlgWorkspaceCpu::getScaleAddIBuffer()
  {
    return scaleaddi_buffer_;
  }

  void LinAlgWorkspaceCpu::setScaleAddIBuffer(ScaleAddIBuffer* buffer)
  {
    assert(scaleaddi_buffer_ == nullptr);
    scaleaddi_buffer_ = buffer;
  }

} // namespace ReSolve
