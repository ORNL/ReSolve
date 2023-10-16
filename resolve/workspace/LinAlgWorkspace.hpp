#pragma once


#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspace
  {
    public:
      LinAlgWorkspace();
      ~LinAlgWorkspace();
      void initializeHandles();
    protected:
      MemoryHandler mem_;
  };

}
