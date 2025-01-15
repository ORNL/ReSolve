/**
 * @file HykktPermutationTests.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @brief Tests for class hykkt::Permutation
 * 
 */
#include <string>
#include <iostream>
#include <fstream>

#include "tests/unit/hykkt/HykktPermutationTests.hpp"

int main(int, char**)
{
  ReSolve::tests::HykktPermutationTests test;

  ReSolve::tests::TestingResults result;
  result += test.permutation();
  return result.summary();
}
