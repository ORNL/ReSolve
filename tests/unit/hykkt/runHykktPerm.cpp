#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "tests/unit/hykkt/HykktPerm.hpp"

int main(int, char**)
{
  ReSolve::tests::HykktPerm test;

  ReSolve::tests::TestingResults result;
  result += test.permutation();
  return result.summary();
}