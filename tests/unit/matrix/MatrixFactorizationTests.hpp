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
    // status.skipTest();

    ReSolve::LinSolverDirectCpuILU0* solver = new ReSolve::LinSolverDirectCpuILU0();
    ReSolve::matrix::Csr* A = createCsrMatrix(0, "cpu");
    solver->setup(A);

    status *= verifyAnswer(*(solver->getLFactor()), rowsL_, colsL_, valsL_, "cpu");
    status *= verifyAnswer(*(solver->getUFactor()), rowsU_, colsU_, valsU_, "cpu");
    
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

  bool verifyAnswer(matrix::Sparse& A,
                    const std::vector<index_type>& answer_rows,
                    const std::vector<index_type>& answer_cols,
                    const std::vector<real_type>&  answer_vals,
                    std::string memspace)
  {
    bool status = true;
    if (memspace != "cpu") {
      A.copyData(memory::DEVICE);
    }

    size_t N = static_cast<size_t>(A.getNumRows());
    for (size_t i = 0; i <= N; ++i) {
      if (A.getRowData(memory::HOST)[i] != answer_rows[i]) {
        status = false;
        std::cout << "Matrix row pointer rows[" << i << "] = " << A.getRowData(memory::HOST)[i]
                  << ", expected: " << answer_rows[i] << "\n";
      }
    }

    size_t NNZ = static_cast<size_t>(A.getNnz());
    for (size_t i = 0; i < NNZ; ++i) {
      if (A.getColData(memory::HOST)[i] != answer_cols[i]) {
        status = false;
        std::cout << "Matrix column index cols[" << i << "] = " << A.getColData(memory::HOST)[i]
                  << ", expected: " << answer_cols[i] << "\n";
      }
      if (!isEqual(A.getValues(memory::HOST)[i], answer_vals[i])) {
        status = false;
        std::cout << "Matrix value element vals[" << i << "] = " << A.getValues(memory::HOST)[i]
                  << ", expected: " << answer_vals[i] << "\n";
        // break; 
      }
    }
    return status;
  }

  std::vector<index_type> rowsL_ = {0, 0, 0, 1, 1, 2, 4, 6, 7, 9};
  std::vector<index_type> colsL_ = {0, 0, 1, 3, 2, 4, 0, 2, 4};
  std::vector<real_type>  valsL_ = {0.5,
                                          0.5,
                                          5.714285714285714e-01,
                                          7.142857142857144e-01,
                                          20.0,
                                          120.0,
                                          1.0,
                                          70.0,
                                          417.5};

  //  (3,1)       0.5000
  //  (5,1)       0.5000
  //  (6,2)       0.5714
  //  (6,4)       0.7143
  //  (7,3)      20.0000
  //  (7,5)     120.0000
  //  (8,1)       1.0000
  //  (9,3)      70.0000
  //  (9,5)     417.5000

  std::vector<index_type> rowsU_ = {0, 3, 6, 9, 12, 13, 15, 17, 19, 20};
  std::vector<index_type> colsU_ = {0, 4, 6, 1, 3, 5, 2, 4, 7, 3, 5, 8, 4, 5, 6, 6, 7, 7, 8, 8};
  std::vector<real_type>  valsU_ = {2.0,
                                    1.0,
                                    3.0,
                                    7.0,
                                    5.0,
                                    4.0,
                                    0.1,
                                    2.5,
                                    2.0,
                                    3.0,
                                    2.0,
                                    8.0,
                                   -0.4,
                                   -2.714285714285714,
                                    6.0,
                                    3.0,
                                  -37.0,
                                    5.0,
                                    1.0,
                                    4.0};
  //  (1,1)       2.0000
  //  (1,5)       1.0000
  //  (1,7)       3.0000
  //  (2,2)       7.0000
  //  (2,4)       5.0000
  //  (2,6)       4.0000
  //  (3,3)       0.1000
  //  (3,5)       2.5000
  //  (3,8)       2.0000
  //  (4,4)       3.0000
  //  (4,6)       2.0000
  //  (4,9)       8.0000
  //  (5,5)      -0.4000
  //  (6,6)      -2.7143
  //  (6,7)       6.0000
  //  (7,7)       3.0000
  //  (7,8)     -37.0000
  //  (8,8)       5.0000
  //  (8,9)       1.0000
  //  (9,9)       4.0000


  matrix::Csr* createCsrMatrix(const index_type /* k */, std::string memspace)
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

    const index_type N   = static_cast<index_type>(rows.size() - 1);
    const index_type NNZ = static_cast<index_type>(cols.size()); 
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
