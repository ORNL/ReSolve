#include <string>
#include <iostream>
#include <fstream>
#include "GramSchmidtTests.hpp"

int main(int argc, char* argv[])
{
  ReSolve::tests::TestingResults result; 

  {
    std::cout << "Running tests with CUDA backend:\n";
    ReSolve::tests::GramSchmidtTests test("cuda");

    result += test.orthogonalize(5000, ReSolve::mgs);
    result += test.orthogonalize(5000, ReSolve::cgs2);
    result += test.orthogonalize(5000, ReSolve::mgs_two_synch);
    result += test.orthogonalize(5000, ReSolve::mgs_pm);
    std::cout << "\n";
  }

  return result.summary();
}
