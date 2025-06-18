#include <fstream>
#include <iostream>
#include <string>

#include "MemoryUtilsTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

#ifdef RESOLVE_USE_HIP
  {
    std::cout << "Running memory tests with HIP backend:\n";
    ReSolve::tests::MemoryUtilsTests test("hip");

    result += test.allocateAndDelete();
    result += test.memsetAndMemcpy();

    std::cout << "\n";
  }
#endif

#ifdef RESOLVE_USE_CUDA
  {
    std::cout << "Running memory tests with CUDA backend:\n";
    ReSolve::tests::MemoryUtilsTests test("hip");

    result += test.allocateAndDelete();
    result += test.memsetAndMemcpy();

    std::cout << "\n";
  }
#endif

  return result.summary();
}
