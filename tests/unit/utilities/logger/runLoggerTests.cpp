/**
 * @file runLoggerTests.cpp
 * @brief Driver for Logger class tests.
 * @author Slaven Peles <peless@ornl.org>
 */

#include <iostream>
#include <ostream>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/Common.hpp>
#include "LoggerTests.hpp"

int main()
{
  using namespace ReSolve::io;

  // Create LoggerTests object
  ReSolve::tests::LoggerTests test;

  // Create test results accounting object
  ReSolve::tests::TestingResults result;

  // Run tests
  result += test.errorOutput();
  result += test.warningOutput();
  result += test.summaryOutput();
  result += test.miscOutput();

  // Return tests summary
  return result.summary();
}
