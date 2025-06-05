#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include "SparseTests.hpp"

/**
 * @brief Run sparse matrix tests with a given backend
 *
 * @tparam WorkspaceType workspace type LinAlgWorkspace{Cpu, CUDA, HIP} supported
 * @param[in] backend - name of the hardware backend
 * @param[out] result - test results
 */
template<typename WorkspaceType>
void runTests(const std::string& backend, ReSolve::memory::MemorySpace memspace, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running tests on " << backend << ":\n";

  WorkspaceType workspace;
  workspace.initializeHandles();

  ReSolve::tests::SparseTests test(memspace);

  ReSolve::io::Logger::setVerbosity(ReSolve::io::Logger::NONE);

  result += test.constructor(50, 50, 2);
  result += test.constructor(50, 100, 2);
  result += test.constructor(50, 50, 2, true);
  result += test.constructor(50, 50, 2, false, false);

  result += test.setDataPointers(50, 50, 2);
  result += test.setDataPointers(50, 100, 2);
  result += test.setValuesPointer(50, 50, 2);
  result += test.setValuesPointer(50, 100, 2);

  result += test.copyValues(50, 50, 2);
  result += test.copyValues(50, 100, 2);
  result += test.copyValuesAndSetValues(50, 50, 2);
  result += test.copyValuesAndSetValues(50, 100, 2);
  result += test.copyValuesAndSetDataPointers(50, 50, 2);
  result += test.copyValuesAndSetDataPointers(50, 100, 2);

  result += test.allocateAndDestroyData(50, 50, 2);
  result += test.allocateAndDestroyData(50, 100, 2);
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