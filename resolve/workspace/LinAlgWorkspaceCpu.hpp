#pragma once

#include <random>

namespace ReSolve
{
  class LinAlgWorkspaceCpu
  {
  public:
    LinAlgWorkspaceCpu();
    ~LinAlgWorkspaceCpu();
    void initializeHandles();
    void resetLinAlgWorkspace();
    std::mt19937& getRng();

  private:
    std::mt19937 generator_;
  };

} // namespace ReSolve
