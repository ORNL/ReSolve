#include <string>
#include <iostream>
#include <fstream>
#include "VectorHandlerTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result; 

  {
    std::cout << "Running tests on CPU:\n";
    ReSolve::tests::VectorHandlerTests test("cpu");
      
    result += test.vectorHandlerConstructor();
    result += test.dot(50);
    result += test.axpy(50);
    result += test.scal(50);
    result += test.infNorm(50);

    std::cout << "\n";
  }

#ifdef RESOLVE_USE_CUDA
  {
    std::cout << "Running tests with CUDA backend:\n";
    ReSolve::tests::VectorHandlerTests test("cuda");

    result += test.dot(5000);
    result += test.axpy(5000);
    result += test.scal(5000);
    result += test.gemv(5000, 10);
    result += test.massAxpy(100, 10);
    result += test.massAxpy(1000, 30);
    result += test.massDot(100, 10);
    result += test.massDot(1000, 30);
    result += test.infNorm(1000);

    std::cout << "\n";
  }
#endif

#ifdef RESOLVE_USE_HIP
  {
    std::cout << "Running tests with HIP backend:\n";
    ReSolve::tests::VectorHandlerTests test("hip");

    result += test.dot(5000);
    result += test.axpy(5000);
    result += test.scal(5000);
    result += test.gemv(5000, 10);
    result += test.massAxpy(100, 10);
    result += test.massAxpy(1000, 300);
    result += test.massDot(100, 10);
    result += test.massDot(1000, 30);
    result += test.infNorm(1000);

    std::cout << "\n";
  }
#endif
  return result.summary();
}
