#include <iostream>

#include "MatrixConversionTests.hpp"

int main()
{
  ReSolve::tests::TestingResults result;
  ReSolve::tests::MatrixConversionTests test;

  result += test.simpleUpperUnexpandedSymmetricMatrix();
  result += test.simpleLowerUnexpandedSymmetricMatrix();
  result += test.simpleMainDiagonalOnlyMatrix();
  result += test.simpleFullAsymmetricMatrix();

  std::cout << std::endl;

  return result.summary();
}
