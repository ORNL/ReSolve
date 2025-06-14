#include <string>
#include <iostream>
#include <fstream>
#include "VectorTests.hpp"


int main(int, char**)
{
  ReSolve::tests::TestingResults result;

  {
    ReSolve::tests::VectorTests test;

    std::cout << "Running tests on CPU:\n";
    result += test.vectorConstructor(50, 5);
    result += test.vectorConstructor(50);
    
    result += test.setData(50);
    
    result += test.copyDataFrom(50);
    result += test.copyDataTo(50);

    result += test.resize(100, 50);
    
    result += test.setToConst(50, 0.0);
    result += test.setToConst(50, 5.0);
  }

#ifdef RESOLVE_USE_GPU
  {
    ReSolve::tests::VectorTests test(ReSolve::memory::DEVICE);

    std::cout << "Running tests on GPU:\n";
    result += test.vectorConstructor(50, 5);
    result += test.vectorConstructor(50);
    
    result += test.setData(50);
    
    result += test.copyDataFrom(50);
    // result += test.copyDataTo(50);

    result += test.resize(100, 50);
    
    // result += test.setToConst(50, 0.0);
    // result += test.setToConst(50, 5.0);
    result += test.syncData(50, ReSolve::memory::HOST);
    result += test.syncData(50, ReSolve::memory::DEVICE);
  }
#endif


  return result.summary();
}
