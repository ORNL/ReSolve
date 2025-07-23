/**
 * @file runHykktCholeskyTests.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Tests for class hykkt::CholeskySolver
 *
 */
#include <fstream>
#include <iostream>
#include <string>

#include "tests/unit/hykkt/HykktCholeskyTests.hpp"

/**
 * @brief Run tests with a given backend
 *
 * @param backend - string name of the hardware backend
 * @param result - test results
 */
void runTests(const std::string& backend, ReSolve::memory::MemorySpace memspace, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running tests on " << backend << " device:\n";

  ReSolve::tests::HykktCholeskyTests test(memspace);

  result += test.test();

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;
  runTests("CPU", ReSolve::memory::HOST, result);

#ifdef RESOLVE_USE_CUDA
  runTests("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
