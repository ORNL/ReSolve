/**
 * @file HykktPermutationTests.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Tests for class hykkt::Permutation
 * 
 */
#include <string>
#include <iostream>
#include <fstream>

#include "tests/unit/hykkt/HykktPermutationTests.hpp"

/** 
 * @brief Run tests with a given backend
 * 
 * @param backend - string name of the hardware backend
 * @param result - test results
 */
void runTests(const std::string& backend, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running tests on " << backend << " device:\n";

  ReSolve::tests::HykktPermutationTests test(backend);

  result += test.permutation();
  
  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;
  runTests("CPU", result);

#ifdef RESOLVE_USE_CUDA
  runTests("CUDA", result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests("HIP", result);
#endif
  
  return result.summary();
}