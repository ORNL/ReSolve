#include <string>
#include <iostream>
#include <fstream>
#include "VectorTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  {
    ReSolve::tests::VectorTests test;

    std::cout << "Running tests on CPU:\n";
    result += test.vectorConstructor();
    result += test.vectorSetToConst();
  }

#ifdef RESOLVE_USE_GPU
  {
    ReSolve::tests::VectorTests test(ReSolve::memory::DEVICE);

    std::cout << "Running tests on GPU:\n";
    result += test.vectorConstructor();
    result += test.vectorSetToConst();
    result += test.vectorSyncData();
  }
#endif


  return result.summary();
}
