#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixHandlerTests.hpp"

int main(int argc, char* argv[])
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

  {
    std::cout << "Running tests with CUDA backend:\n";
    ReSolve::tests::MatrixHandlerTests test("cuda");

    result += test.matrixHandlerConstructor();
    result += test.matrixOneNorm();
    result += test.matVec(50);

    std::cout << "\n";
  }

  return result.summary();
}