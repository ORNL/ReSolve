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

    ReSolve::tests::VectorTests test(handler);
    result += test.vectorConstructor();
    result += test.dimensions(50, 5);
    result += test.setData(50);

    std::cout << "\n";
  }

  return result.summary();
}