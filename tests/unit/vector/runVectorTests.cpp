#include <string>
#include <iostream>
#include <fstream>
#include "VectorTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result; 

  {
    std::cout << "Running tests on CPU:\n";

    ReSolve::LinAlgWorkspaceCpu workspace;
    workspace.initializeHandles();
    ReSolve::VectorHandler handler(&workspace);

    ReSolve::tests::VectorTests test;
    result += test.vectorConstructor(50, 5);
    result += test.vectorConstructor(50);
    result += test.setData(50);
    result += test.copyDataFromArray(50);
    result += test.copyDataFromVector(50);
    result += test.copyDataToArray(50);
    result += test.copyDataToVector(50);
    result += test.resize(100, 50);
    result += test.syncData(50, ReSolve::memory::HOST);
    result += test.syncData(50, ReSolve::memory::DEVICE);

    std::cout << "\n";
  }

  return result.summary();
}