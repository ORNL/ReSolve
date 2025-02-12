#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixHandlerTests.hpp"

#if defined (RESOLVE_USE_CUDA)
  ReSolve::LinAlgWorkspaceCUDA workspace;
#elif defined (RESOLVE_USE_HIP)
  ReSolve::LinAlgWorkspaceHIP workspace;
#else
  ReSolve::LinAlgWorkspaceCpu workspace;
#endif

int main(int, char**)
{
  #if defined (RESOLVE_USE_CUDA)
    std::cout << "Running tests with CUDA backend:\n";
  #elif defined (RESOLVE_USE_HIP)
    std::cout << "Running tests with HIP backend:\n";
  #else
    std::cout << "Running tests on CPU:\n";
  #endif
  ReSolve::tests::TestingResults result; 
  workspace.initializeHandles();
  ReSolve::MatrixHandler handler(&workspace);
  ReSolve::tests::MatrixHandlerTests test(handler);
  result += test.matrixHandlerConstructor();
  result += test.matrixInfNorm(1000000);
  result += test.matVec(50);
  result += test.csc2csr(1024, 1024);
  result += test.csc2csr(1024, 2048);
  result += test.csc2csr(2048, 1024);
  std::cout << "\n";
  return result.summary();
}

