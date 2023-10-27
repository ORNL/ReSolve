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
    ReSolve::tests::MatrixHandlerTests test("cpu");
      
    result += test.matrixHandlerConstructor();
    result += test.matrixOneNorm();
    result += test.matVec(50);

    std::cout << "\n";
  }

#ifdef RESOLVE_USE_CUDA
  {
    std::cout << "Running tests with CUDA backend:\n";
    ReSolve::tests::MatrixHandlerTests test("cuda");

    result += test.matrixHandlerConstructor();
    result += test.matrixOneNorm();
    result += test.matVec(50);

    std::cout << "\n";
  }
#endif

#ifdef RESOLVE_USE_HIP
  {
    std::cout << "Running tests with HIP backend:\n";
    ReSolve::tests::MatrixHandlerTests test("hip");

    result += test.matrixHandlerConstructor();
    result += test.matrixOneNorm();
    result += test.matVec(50);

    std::cout << "\n";
  }
#endif
  return result.summary();
}
