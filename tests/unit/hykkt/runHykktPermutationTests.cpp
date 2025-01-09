/**
 * @file runHykktPermutationTests.cpp
 * @author your name (you@domain.com)
 * @brief 
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