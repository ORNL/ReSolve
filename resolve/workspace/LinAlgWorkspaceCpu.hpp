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
      // MemoryHandler mem_; ///< Memory handler not needed for now
  };

}
