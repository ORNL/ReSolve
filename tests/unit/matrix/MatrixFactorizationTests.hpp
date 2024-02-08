/**
 * @file MatrixFactorizationTests.hpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Class with matrix factorization unit tests
 * 
 */
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { namespace tests {

/**
 * @class Unit tests for matrix factorization
 */
class MatrixFactorizationTests : TestBase
{
public:
  MatrixFactorizationTests(std::string memspace) : memspace_(memspace) 
  {}
  virtual ~MatrixFactorizationTests()
  {}

  TestOutcome matrixFactorizationConstructor()
  {
    TestStatus status;
    status.skipTest();

    ReSolve::LinSolverDirectCpuILU0* solver = new ReSolve::LinSolverDirectCpuILU0();
    ReSolve::matrix::Csr* A = createCsrMatrix(0, "cpu");
    solver->setup(A);

    delete A;
    delete solver;
    
    return status.report(__func__);
  }

  TestOutcome matrixILU0()
  {
    TestStatus status;
    status.skipTest();
    
    return status.report(__func__);
  }

  TestOutcome matrixInfNorm(index_type N)
  {
    TestStatus status;
    ReSolve::memory::MemorySpace ms;
    if (memspace_ == "cpu")
      ms = memory::HOST;
    else
      ms = memory::DEVICE;

    ReSolve::MatrixHandler* handler = createMatrixHandler();

    matrix::Csr* A = createCsrMatrix(N, memspace_);
    real_type norm;
    handler->matrixInfNorm(A, &norm, ms);
    status *= (norm == 30.0); 
    
    delete handler;
    delete A;
    return status.report(__func__);
  }


private:
  std::string memspace_{"cpu"};

  ReSolve::MatrixHandler* createMatrixHandler()
  {
    if (memspace_ == "cpu") {
      LinAlgWorkspaceCpu* workspace = new LinAlgWorkspaceCpu();
      return new MatrixHandler(workspace);
#ifdef RESOLVE_USE_CUDA
    } else if (memspace_ == "cuda") {
      LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
      workspace->initializeHandles();
      return new MatrixHandler(workspace);
#endif
#ifdef RESOLVE_USE_HIP
    } else if (memspace_ == "hip") {
      LinAlgWorkspaceHIP* workspace = new LinAlgWorkspaceHIP();
      workspace->initializeHandles();
      return new MatrixHandler(workspace);
#endif
    } else {
      std::cout << "ReSolve not built with support for memory space " << memspace_ << "\n";
    }
    return nullptr;
  }

  bool verifyAnswer(vector::Vector& x, real_type answer, std::string memspace)
  {
    bool status = true;
    if (memspace != "cpu") {
      x.copyData(memory::DEVICE, memory::HOST);
    }

    for (index_type i = 0; i < x.getSize(); ++i) {
      // std::cout << x.getData(memory::HOST)[i] << "\n";
      if (!isEqual(x.getData(memory::HOST)[i], answer)) {
        status = false;
        std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                  << ", expected: " << answer << "\n";
        break; 
      }
    }
    return status;
  }

  matrix::Csr* createCsrMatrix(const index_type k, std::string memspace)
  {
    std::vector<index_type> rows = {0, 3, 6, 9, 12, 13, 17, 21, 24, 27};
    std::vector<index_type> cols = {0, 4, 6,
                                    1, 3, 5,
                                    0, 4, 7,
                                    3, 5, 8,
                                    0,
                                    1, 3, 5, 6,
                                    2, 4, 6, 7,
                                    0, 7, 8,
                                    2, 4, 8};
    std::vector<real_type>  vals = {2., 1., 3., 
                                    7., 5., 4.,
                                    1., 3., 2.,
                                    3., 2., 8.,
                                    1.,
                                    4., 5., 1., 6.,
                                    2., 2., 3., 3.,
                                    2., 5., 1.,
                                    7., 8., 4.};

    const index_type N   = rows.size() - 1;
    const index_type NNZ = cols.size(); 
    // std::cout << N << "\n";

    // Allocate NxN CSR matrix with NNZ nonzeros
    matrix::Csr* A = new matrix::Csr(N, N, NNZ);
    A->allocateMatrixData(memory::HOST);
    A->updateData(&rows[0], &cols[0], &vals[0], memory::HOST, memory::HOST);

    // A->print();

    if ((memspace == "cuda") || (memspace == "hip")) {
      A->copyData(memory::DEVICE);
    }

    return A;
  }
}; // class MatrixFactorizationTests

}} // namespace ReSolve::tests
