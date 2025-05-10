#include <string>
#include <iostream>
#include <fstream>
#include "VectorTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  ReSolve::tests::VectorTests test;

  {
    std::cout << "Running tests on CPU:\n";
    result += test.vectorConstructor();

  }

#ifdef RESOLVE_USE_GPU
  {
    std::cout << "Running tests on GPU:\n";
    result += test.vectorConstructor();

  }
#endif


  return result.summary();
}
