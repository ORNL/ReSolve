#include <string>
#include <iostream>
#include <fstream>
#include "VectorTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  ReSolve::tests::VectorTests tests();

  {
    std::cout << "Running tests on CPU:\n";

  }

#ifdef RESOLVE_USE_GPU
  {
    std::cout << "Running tests on GPU:\n";

  }
#endif


  return result.summary();
}
