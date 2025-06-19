/**
 * @file runParamTests.cpp
 * @brief Driver for Param class tests.
 * @author Slaven Peles <peless@ornl.org>
 */

#include <iostream>
#include <ostream>

#include "ParamTests.hpp"
#include <resolve/Common.hpp>

int main()
{
  using namespace ReSolve::io;

  // Create ParamTests object
  ReSolve::tests::ParamTests test;

  // Create test results accounting object
  ReSolve::tests::TestingResults result;

  // Run tests
  result += test.paramSetGet();

  // Return tests summary
  return result.summary();
}
