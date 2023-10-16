#pragma once


#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class LinAlgWorkspace
  {
    public:
      LinAlgWorkspace();
      ~LinAlgWorkspace();
    protected:
      MemoryHandler mem_;
  };

}
