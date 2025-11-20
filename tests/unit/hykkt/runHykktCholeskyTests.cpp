/**
 * @file runHykktCholeskyTests.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Tests for class hykkt::CholeskySolver
 *
 */
#include <fstream>
#include <iostream>
#include <string>

#include "tests/unit/hykkt/HykktCholeskyTests.hpp"

/**
 * @brief Run tests with a given backend
 *
 * @param backend - string name of the hardware backend
 * @param result - test results
 */
template <typename WorkspaceType>
void runTests(const std::string& backend, ReSolve::memory::MemorySpace memspace, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running tests on " << backend << " device:\n";

  WorkspaceType workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler handler(&workspace);

  ReSolve::tests::HykktCholeskyTests test(memspace, handler);

  result += test.minimalCorrectness();
  handler.setValuesChanged(true, memspace);
  workspace.resetLinAlgWorkspace(); // reset is necessary due to different sparsity.

  for (int size : {3, 10, 100, 1000})
  {
    result += test.randomized(size);
    handler.setValuesChanged(true, memspace);
    workspace.resetLinAlgWorkspace();
  }

  result += test.randomizedReuseSparsityPattern(3, 10);

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;
  runTests<ReSolve::LinAlgWorkspaceCpu>("CPU", ReSolve::memory::HOST, result);

#ifdef RESOLVE_USE_CUDA
  runTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests<ReSolve::LinAlgWorkspaceHIP>("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
