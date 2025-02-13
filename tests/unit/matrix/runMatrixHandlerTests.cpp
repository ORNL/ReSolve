#include <string>
#include <iostream>
#include <fstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include "MatrixHandlerTests.hpp"

template<typename WorkspaceType>
void runTests(const std::string& backendName, ReSolve::tests::TestingResults& result) {
    std::cout << "Running tests with " << backendName << ":\n";
    
    WorkspaceType workspace;
    workspace.initializeHandles();
    ReSolve::MatrixHandler handler(&workspace);

    ReSolve::tests::MatrixHandlerTests test(handler);
    result += test.matrixHandlerConstructor();
    result += test.matrixInfNorm(1000000);
    result += test.matVec(50);
    result += test.csc2csr(5, 5);
    result += test.csc2csr(5, 3);
    result += test.csc2csr(3, 5);
    result += test.csc2csr(1024, 1024);
    result += test.csc2csr(1024, 2048);
    result += test.csc2csr(2048, 1024);
    result += test.csc2csr(1024, 1200);
    result += test.csc2csr(1200, 1024);
    std::cout << "\n";
}

int main(int, char**) {
    ReSolve::tests::TestingResults result;

    runTests<ReSolve::LinAlgWorkspaceCpu>("CPU", result);

#ifdef RESOLVE_USE_CUDA
    runTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA backend", result);
#endif

#ifdef RESOLVE_USE_HIP
    runTests<ReSolve::LinAlgWorkspaceHIP>("HIP backend", result);
#endif

    return result.summary();
}

