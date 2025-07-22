#pragma once

namespace ReSolve
{
  class LinAlgWorkspaceCpu
  {
  public:
    LinAlgWorkspaceCpu();
    ~LinAlgWorkspaceCpu();
    void initializeHandles();
    void resetLinAlgWorkspace();
  };

} // namespace ReSolve
