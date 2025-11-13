/**
 * @file runHykktRuizScalingTests.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Tests for class hykkt::RuizScaling
 *
 */
#include <fstream>
#include <iostream>
#include <string>

#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif
#ifdef RESOLVE_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

#include "HykktRuizScalingTests.hpp"

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

  ReSolve::tests::HykktRuizScalingTests test(memspace, handler);

  int test_values[] = {8, 9, 10, 68, 256, 512, 1024, 2048, 4096};

  for (int n : test_values)
  {
    result += test.ruizTest(n);
    workspace.resetLinAlgWorkspace();
  }

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
