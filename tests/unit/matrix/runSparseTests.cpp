#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
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
void runTests(const std::string& backend, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running tests on " << backend << " device:\n";

  WorkspaceType workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler handler(&workspace);

  ReSolve::tests::SparseTests test(handler);

  result += test.constructor(50, 50, 100);
  result += test.constructor(50, 50, 100, true);
  result += test.constructor(50, 50, 100, false, false);

  result += test.setDataPointers(50, 50, 100);
  result += test.setValuesPointer(50, 50, 100);

  result += test.allocateAndDestroyData(50, 50, 100);
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