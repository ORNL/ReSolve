#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixIoTests.hpp"

int main(int argc, char* argv[])
{
  ReSolve::tests::MatrixIoTests test;

  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";

  ReSolve::tests::TestingResults result;
  result += test.cooMatrixImport();
  result += test.cooMatrixReadAndUpdate();
  result += test.rhsVectorReadFromFile();
  result += test.rhsVectorReadAndUpdate();

  return result.summary();
}