#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixIoTests.hpp"

int main(int, char**)
{
  ReSolve::tests::MatrixIoTests test;

  ReSolve::tests::TestingResults result;
  result += test.cooMatrixImport();
  result += test.csrMatrixImport();
  result += test.cooMatrixExport();
  result += test.csrMatrixExport();
  result += test.cooMatrixReadAndUpdate();
  result += test.csrMatrixReadAndUpdate();
  result += test.rhsArrayReadFromFile();
  result += test.rhsVectorReadFromFile();
  result += test.rhsArrayReadAndUpdate();
  result += test.rhsVectorReadAndUpdate();

  return result.summary();
}