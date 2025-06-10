#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixHandlerTests.hpp"


/**
 * @brief run tests with a given backend
 *
 * Checks correctness of the constructor, matrixInfNorm, matVec, csc2csr
 * CPU, CUDA and HIP backends are supported
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

  ReSolve::tests::MatrixHandlerTests test(handler);
  result += test.matrixHandlerConstructor();
  result += test.matrixInfNorm(10000);
  result += test.matVec(50);
  result += test.csc2csr(3, 3);
  result += test.csc2csr(5, 3);
  result += test.csc2csr(3, 5);
  result += test.csc2csr(1024, 1024);
  result += test.csc2csr(1024, 2048);
  result += test.csc2csr(2048, 1024);
  result += test.csc2csr(1024, 1200);
  result += test.csc2csr(1200, 1024);
  result += test.transpose(3, 3);
  result += test.transpose(5, 3);
  result += test.transpose(3, 5);
  result += test.transpose(1024, 1024);
  result += test.transpose(1024, 2048);
  result += test.transpose(2048, 1024);
  result += test.transpose(1024, 1200);
  result += test.transpose(1200, 1024);
  result += test.leftScale(1024, 1024);
  result += test.leftScale(1024, 2048);
  result += test.leftScale(2048, 1024);
  result += test.rightScale(1024, 1024);
  result += test.rightScale(1024, 2048);
  result += test.rightScale(2048, 1024);
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

