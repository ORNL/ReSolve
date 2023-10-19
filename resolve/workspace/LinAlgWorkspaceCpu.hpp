#pragma once


#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspaceCpu
  {
    public:
      LinAlgWorkspaceCpu();
      ~LinAlgWorkspaceCpu();
      void initializeHandles();
    private:
      MemoryHandler mem_;
  };

}
