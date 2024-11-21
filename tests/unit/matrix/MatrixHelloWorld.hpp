#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <tests/unit/TestBase.hpp>
#include <resolve/matrix/Coo.hpp>

namespace ReSolve { namespace tests {

class MatrixHelloWorld : TestBase
{
public:
  MatrixHelloWorld(){}
  virtual ~MatrixHelloWorld(){}


  TestOutcome addition()
  {
    real_type a = 1.0;
    real_type b = 2.0;
    real_type c = a + b;
    if (isEqual(c, 3.0))
      return PASS;
    else
      return FAIL;
  }

  TestOutcome subtraction()
  {
    real_type a = 1.0;
    real_type b = 2.0;
    real_type c = a - b;
    if (isEqual(c, -1.0))
      return PASS;
    else
      return FAIL;
  }

  TestOutcome multiplication()
  {
    real_type a = 1.0;
    real_type b = 2.0;
    real_type c = a * b;
    if (isEqual(c, 2.0))
      return PASS;
    else
      return FAIL;
  }

  TestOutcome division()
  {
    real_type a = 1.0;
    real_type b = 2.0;
    real_type c = a / b;
    if (isEqual(c, 0.5))
      return PASS;
    else
      return FAIL;
  }

  TestOutcome helloWorld()
  {
    std::cout << "Hello World!" << std::endl;
    return PASS;
  }

  

   
}; // class MatrixHelloWorld

}} // namespace ReSolve::tests
