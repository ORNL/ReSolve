/**
 * @file runPreconditionerIdentityTests.cpp
 * @brief Tests for PreconditionerIdentity class.
 *
 */

#include <iostream>
#include <string>

#include "PreconditionerIdentityTests.hpp"

/**
 * @brief Run PreconditionerIdentity tests in a given memory space.
 */
void runTests(const std::string&              backend,
              ReSolve::memory::MemorySpace    memspace,
              ReSolve::tests::TestingResults& result)
{
  std::cout << "Running PreconditionerIdentity tests on " << backend << ":\n";

  ReSolve::tests::PreconditionerIdentityTests test(memspace);

  result += test.setup();
  result += test.checkSide();
  result += test.apply();

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  runTests("CPU", ReSolve::memory::HOST, result);

#if defined(RESOLVE_USE_CUDA) || defined(RESOLVE_USE_HIP)
  runTests("GPU", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
