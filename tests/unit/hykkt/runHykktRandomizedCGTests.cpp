/**
 * @file runHykktRandomizedCGTests.hpp
 * @brief Tests for class hykkt::SchurComplementConjugateGradient
 *
 */
#include <fstream>
#include <iostream>
#include <string>

#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif
#ifdef RESOLVE_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

#include "HykktRandomizedCGTests.hpp"
#include <resolve/vector/Vector.hpp>

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
  ReSolve::MatrixHandler                                     matrix_handler(&workspace);
  ReSolve::VectorHandler                                     vector_handler(&workspace);
  ReSolve::tests::HykktRandomizedConjugateGradientTests test(memspace, matrix_handler, vector_handler);

  std::string        source_dir = std::string(SOURCE_DIR);
  std::string        A_file_name = source_dir + std::string("/RandomizedCGTestMatrices/crankseg_2.mtx");
  double b_min = -100.0;
  double b_max = 100.0;

  result += test.RandomizedCGTest(A_file_name, 1, b_min, b_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedCGTest(A_file_name, 2, b_min, b_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedCGTest(A_file_name, 4, b_min, b_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedCGTest(A_file_name, 8, b_min, b_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedCGTest(A_file_name, 16, b_min, b_max);

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;
  // runTests<ReSolve::LinAlgWorkspaceCpu>("CPU", ReSolve::memory::HOST, result);

#ifdef RESOLVE_USE_CUDA
  runTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests<ReSolve::LinAlgWorkspaceHIP>("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
