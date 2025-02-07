#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixHandlerTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result; 

  {
    std::cout << "Running tests on CPU:\n";

    ReSolve::LinAlgWorkspaceCpu workspace;
    workspace.initializeHandles();
    ReSolve::MatrixHandler handler(&workspace);

    ReSolve::tests::MatrixHandlerTests test(handler);
    result += test.matrixHandlerConstructor();
    result += test.matrixInfNorm(10000);
    result += test.matVec(50);

    std::cout << "\n";
  }

#ifdef RESOLVE_USE_CUDA
  {
    std::cout << "Running tests with CUDA backend:\n";
    ReSolve::LinAlgWorkspaceCUDA workspace;
    workspace.initializeHandles();
    ReSolve::MatrixHandler handler(&workspace);

    ReSolve::tests::MatrixHandlerTests test(handler);
    result += test.matrixHandlerConstructor();
    result += test.matrixInfNorm(1000000);
    result += test.matVec(50);
    result += test.csc2csr(2,3);
    result += test.csc2csr(3,2);
    result += test.csc2csr(3,3);
    std::cout << "\n";
  }
#endif

#ifdef RESOLVE_USE_HIP
  {
    std::cout << "Running tests with HIP backend:\n";
    ReSolve::LinAlgWorkspaceHIP workspace;
    workspace.initializeHandles();
    ReSolve::MatrixHandler handler(&workspace);

    ReSolve::tests::MatrixHandlerTests test(handler);
    result += test.matrixHandlerConstructor();
    result += test.matrixInfNorm(1000000);
    result += test.matVec(50);

    std::cout << "\n";
  }
#endif
  return result.summary();
}

