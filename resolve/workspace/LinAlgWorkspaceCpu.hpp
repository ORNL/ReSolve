#pragma once

namespace ReSolve
{
  class LinAlgWorkspaceCpu
  {
  public:
    LinAlgWorkspaceCpu();
    ~LinAlgWorkspaceCpu();
    int initializeHandles();
    void resetLinAlgWorkspace();
  };

} // namespace ReSolve
