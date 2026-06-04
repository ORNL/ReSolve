/**
 * @file runPreconditionerUserMatrixTests.cpp
 * @brief Tests for PreconditionerUserMatrix class.
 *
 */

#include <iostream>
#include <string>

#include "PreconditionerUserMatrixTests.hpp"
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

/**
 * @brief Run PreconditionerUserMatrix tests with a given backend.
 *
 */
template <typename WorkspaceType>
void runExplicitMatrixTests(const std::string&              backend,
                            ReSolve::memory::MemorySpace    memspace,
                            ReSolve::tests::TestingResults& result)
{
  std::cout << "Running PreconditionerUserMatrix tests on " << backend << ":\n";

  WorkspaceType workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler handler(&workspace);

  ReSolve::tests::PreconditionerUserMatrixTests test(memspace, handler);

  result += test.checkSide();
  result += test.setPrecMatrix();
  result += test.apply();

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  runExplicitMatrixTests<ReSolve::LinAlgWorkspaceCpu>("CPU", ReSolve::memory::HOST, result);

#ifdef RESOLVE_USE_CUDA
  runExplicitMatrixTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runExplicitMatrixTests<ReSolve::LinAlgWorkspaceHIP>("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
