#include <string>
#include <iostream>
#include <fstream>
#include "VectorTests.hpp"

/**
 * @brief Run vector tests with a given backend
 *
 * @tparam WorkspaceType workspace type LinAlgWorkspace{Cpu, CUDA, HIP} supported
 * @param[in] backend - name of the hardware backend
 * @param[out] result - test results
 */
template<typename WorkspaceType>
void runTests(const std::string& backend, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running tests on " << backend << " device:\n";

  ReSolve::LinAlgWorkspaceCpu workspace;
  workspace.initializeHandles();
  ReSolve::VectorHandler handler(&workspace);

  ReSolve::tests::VectorTests test(handler);
  
  result += test.vectorConstructor(50, 5);
  result += test.vectorConstructor(50);
  
  result += test.setData(50);
  
  result += test.copyDataFrom(50);
  result += test.copyDataTo(50);

  result += test.resize(100, 50);
  
  result += test.setToConst(50, 0.0);
  result += test.setToConst(50, 5.0);

  if (backend != "CPU") {
    result += test.syncData(50, ReSolve::memory::HOST);
    result += test.syncData(50, ReSolve::memory::DEVICE);
  }

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;
  runTests<ReSolve::LinAlgWorkspaceCpu>("CPU", result);

#ifdef RESOLVE_USE_CUDA
  runTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA", result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests<ReSolve::LinAlgWorkspaceHIP>("HIP", result);
#endif

  return result.summary();
}