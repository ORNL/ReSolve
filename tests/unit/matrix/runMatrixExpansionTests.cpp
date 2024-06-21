#include <fstream>
#include <iostream>
#include <string>

#include "MatrixExpansionTests.hpp"

#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>

int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  {
    std::cout << "Running tests on CPU:\n";
    ReSolve::tests::MatrixExpansionTests test;

    result += test.cooMatrix();
    result += test.csrMatrix();
    result += test.cscMatrix();

    std::cout << "\n";
  }

  return result.summary();
}
