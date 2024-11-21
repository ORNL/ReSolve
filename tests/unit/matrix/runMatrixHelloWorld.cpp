#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixHelloWorld.hpp"

int main(int, char**)
{
  ReSolve::tests::MatrixHelloWorld test;

  ReSolve::tests::TestingResults result;
  result += test.addition();
  result += test.subtraction();
  result += test.multiplication();
  result += test.division();
  result += test.helloWorld();
  return result.summary();
}