#include <string>
#include <iostream>
#include <fstream>
#include "GramSchmidtTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result; 

#ifdef RESOLVE_USE_CUDA
  {
    std::cout << "Running tests with CUDA backend:\n";
    ReSolve::tests::GramSchmidtTests test("cuda");

    result += test.GramSchmidtConstructor();
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs2);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_two_sync);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_pm);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs1);
    std::cout << "\n";
  }
#endif

#ifdef RESOLVE_USE_HIP
  {
    std::cout << "Running tests with HIP backend:\n";
    ReSolve::tests::GramSchmidtTests test("hip");

    result += test.GramSchmidtConstructor();
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs2);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_two_sync);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_pm);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs1);
    std::cout << "\n";
  }
#endif

  {
    std::cout << "Running tests on the CPU:\n";
    ReSolve::tests::GramSchmidtTests test("cpu");

    result += test.GramSchmidtConstructor();
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs2);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_two_sync);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_pm);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs1);
    std::cout << "\n";
  }
  return result.summary();
}
