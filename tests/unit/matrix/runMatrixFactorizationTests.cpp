#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixFactorizationTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result; 

  {
    std::cout << "Running tests on CPU:\n";
    ReSolve::tests::MatrixFactorizationTests test("cpu");
      
    result += test.matrixFactorizationConstructor();
    result += test.matrixILU0();

    std::cout << "\n";
  }


  return result.summary();
}
