#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixConversionTests.hpp"

int main(int, char**)
{
  ReSolve::tests::MatrixConversionTests test;

  ReSolve::tests::TestingResults result;
  result += test.cooToCsr();

  return result.summary();
}