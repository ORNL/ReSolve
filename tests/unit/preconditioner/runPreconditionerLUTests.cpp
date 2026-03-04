/**
 * @file runPreconditionerLUTests.cpp
 * @brief Tests for PreconditionerLU class.
 *
 */

#include <iostream>
#include <string>

#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
#endif

#include "PreconditionerLUTests.hpp"

/**
 * @brief Run PreconditionerLU tests with a given backend.
 *
 */
template <typename WorkspaceType, typename ILUSolverType>
void runTests(const std::string&              backend,
              ReSolve::memory::MemorySpace    memspace,
              ReSolve::tests::TestingResults& result)
{
  std::cout << "Running PreconditionerLU tests on " << backend << ":\n";

  WorkspaceType workspace;
  workspace.initializeHandles();
  ILUSolverType ilu_solver(&workspace);

  ReSolve::tests::PreconditionerLUTests test(memspace, &ilu_solver);

  result += test.checkSide();
  result += test.solve();

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  runTests<ReSolve::LinAlgWorkspaceCpu,
           ReSolve::LinSolverDirectCpuILU0>("CPU", ReSolve::memory::HOST, result);

#ifdef RESOLVE_USE_CUDA
  runTests<ReSolve::LinAlgWorkspaceCUDA,
           ReSolve::LinSolverDirectCuSparseILU0>("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests<ReSolve::LinAlgWorkspaceHIP,
           ReSolve::LinSolverDirectRocSparseILU0>("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
