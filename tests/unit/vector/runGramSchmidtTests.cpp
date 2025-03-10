#include <string>
#include <iostream>
#include <fstream>
#include "GramSchmidtTests.hpp"

template <class workspace_type>
static ReSolve::tests::TestingResults runTests();

int main(int, char**)
{
  ReSolve::tests::TestingResults result; 

  std::cout << "\nRunning tests on a CPU device:\n";
  result += runTests<ReSolve::LinAlgWorkspaceCpu>();

#ifdef RESOLVE_USE_CUDA
  std::cout << "\nRunning tests on a CUDA device:\n";
  result += runTests<ReSolve::LinAlgWorkspaceCUDA>();
#endif

#ifdef RESOLVE_USE_HIP
  std::cout << "\nRunning tests on a HIP device:\n";
  result += runTests<ReSolve::LinAlgWorkspaceHIP>();
#endif

  return result.summary();
}

template <class workspace_type>
ReSolve::tests::TestingResults runTests()
{
  ReSolve::tests::TestingResults result; 


  workspace_type workspace;
  workspace.initializeHandles();
  ReSolve::VectorHandler handler(&workspace);

  ReSolve::tests::GramSchmidtTests test(handler);
  result += test.GramSchmidtConstructor();
  result += test.orthogonalize(5000, ReSolve::GramSchmidt::MGS);
  result += test.orthogonalize(5000, ReSolve::GramSchmidt::CGS2);
  result += test.orthogonalize(5000, ReSolve::GramSchmidt::MGS_TWO_SYNC);
  result += test.orthogonalize(5000, ReSolve::GramSchmidt::MGS_PM);
  result += test.orthogonalize(5000, ReSolve::GramSchmidt::CGS1);

  return result;
}
