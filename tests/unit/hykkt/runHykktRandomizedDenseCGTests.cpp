/**
 * @file runHykktRandomizedDenseCGTests.hpp
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

#include "HykktRandomizedDenseCGTests.hpp"
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
  ReSolve::tests::HykktRandomizedDenseConjugateGradientTests test(memspace, matrix_handler, vector_handler);

  size_t n = 1 << 14;
  std::string        source_dir = std::string(SOURCE_DIR);
  std::string        A_file_name = source_dir + std::string("/RandomizedDenseCGTestMatrices/A_") + std::to_string(n) + std::string(".mtx");
  std::string        b_file_name = source_dir + std::string("/RandomizedDenseCGTestMatrices/b_") + std::to_string(n) + std::string(".mtx");
  double rng_min = -1.0;
  double rng_max = 1.0;

  // test.generateVectors(A_file_name, b_file_name, n, rng_min, rng_max);
  // test.choleskyTests(A_file_name, b_file_name, n);
  result += test.RandomizedDenseCGTest(A_file_name, b_file_name, n, 1, rng_min, rng_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedDenseCGTest(A_file_name, b_file_name, n, 2, rng_min, rng_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedDenseCGTest(A_file_name, b_file_name, n, 4, rng_min, rng_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedDenseCGTest(A_file_name, b_file_name, n, 8, rng_min, rng_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedDenseCGTest(A_file_name, b_file_name, n, 16, rng_min, rng_max);
  workspace.resetLinAlgWorkspace();
  result += test.RandomizedDenseCGTest(A_file_name, b_file_name, n, 32, rng_min, rng_max);

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
