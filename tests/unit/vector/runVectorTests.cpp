#include <fstream>
#include <iostream>
#include <string>

#include "VectorTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  {
    ReSolve::tests::VectorTests test;

    std::cout << "Running tests on CPU:\n";
    result += test.vectorConstructor(50, 5);
    result += test.vectorConstructor(50);

    result += test.setData(50);

    result += test.copyDataFrom(50);
    // result += test.copyDataTo(50);

    result += test.resize(100, 50);

    result += test.setToConst(50);
    result += test.setToConst(50);
  }

#ifdef RESOLVE_USE_GPU
  {
    ReSolve::tests::VectorTests test(ReSolve::memory::DEVICE);

    std::cout << "Running tests on GPU:\n";
    result += test.vectorConstructor(50, 5);
    result += test.vectorConstructor(50);

    result += test.setData(50);

    result += test.copyDataFrom(50);
    // result += test.copyDataTo(50);

    result += test.resize(100, 50);

    result += test.setToConst(50);
    result += test.setToConst(50);
    result += test.syncData(50);
    result += test.syncData(50);
  }
#endif

  return result.summary();
}
