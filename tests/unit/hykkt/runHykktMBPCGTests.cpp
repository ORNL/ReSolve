/**
 * @file runHykktMBPCGTests.hpp
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

#include "HykktMBPCGTests.hpp"
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
  std::cout << "Running MBPCG tests on " << backend << " device:\n";

  WorkspaceType workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler                                     matrix_handler(&workspace);
  ReSolve::VectorHandler                                     vector_handler(&workspace);
  ReSolve::tests::HykktMultiBasisParallelConjugateGradientTests<WorkspaceType> test(memspace, matrix_handler, vector_handler, workspace);

  std::string        source_dir = std::string(SOURCE_DIR);

  std::vector<std::string> matrix_names{
    "Fault_639"
  };

  for (const std::string& matrix_name : matrix_names)
  {
    std::string        A_file_name = source_dir + std::string("/MBPCGTestMatrices/") + matrix_name + std::string(".mtx");
    std::string        b_file_name = source_dir + std::string("/MBPCGTestMatrices/") + matrix_name + std::string("_b.mtx");

    printf("\n\n\nMatrix: %s\n", matrix_name.c_str());
    result += test.MBPCGTest(A_file_name, b_file_name, false);

    std::cout << "\n";
  }
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

#ifdef RESOLVE_USE_CUDA
  runTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests<ReSolve::LinAlgWorkspaceHIP>("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
