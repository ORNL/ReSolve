#include <string>
#include <iostream>
#include <fstream>
#include "GramSchmidtTests.hpp"

int main(int, char**)
{
  ReSolve::tests::TestingResults result; 

  {
    std::cout << "Running tests on the CPU:\n";

    ReSolve::LinAlgWorkspaceCpu workspace;
    workspace.initializeHandles();
    ReSolve::VectorHandler handler(&workspace);

    ReSolve::tests::GramSchmidtTests test(handler);
    result += test.GramSchmidtConstructor();
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs2);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_two_sync);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_pm);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs1);
    std::cout << "\n";
  }

#ifdef RESOLVE_USE_CUDA
  {
    std::cout << "Running tests with CUDA backend:\n";

    ReSolve::LinAlgWorkspaceCUDA workspace;
    workspace.initializeHandles();
    ReSolve::VectorHandler handler(&workspace);

    ReSolve::tests::GramSchmidtTests test(handler);
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

    ReSolve::LinAlgWorkspaceHIP workspace;
    workspace.initializeHandles();
    ReSolve::VectorHandler handler(&workspace);

    ReSolve::tests::GramSchmidtTests test(handler);
    result += test.GramSchmidtConstructor();
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs2);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_two_sync);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::mgs_pm);
    result += test.orthogonalize(5000, ReSolve::GramSchmidt::cgs1);
    std::cout << "\n";
  }
#endif

  return result.summary();
}
