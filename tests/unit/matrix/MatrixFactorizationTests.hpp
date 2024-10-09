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
    
    return status.report(__func__);
  }

  /**
   * @brief Test ILU0 factorization and triangular solve.
   * 
   * @return TestOutcome 
   */
  TestOutcome matrixILU0()
  {
    TestStatus status;

    ReSolve::LinSolverDirectCpuILU0 solver;
    ReSolve::matrix::Csr* A = createCsrMatrix(0, "cpu");

    ReSolve::vector::Vector rhs(A->getNumRows());
    rhs.setToConst(constants::ONE, memory::HOST);

    ReSolve::vector::Vector x(A->getNumRows());
    x.allocate(memory::HOST);

    // Reference solutions are for zero diagonal approximated with 0.1
    solver.setZeroDiagonal(0.1);

    // Test ILU0 analysis and factorization
    solver.setup(A);
    status *= verifyAnswer(*(solver.getLFactor()), rowsL_, colsL_, valsL_, "cpu");
    status *= verifyAnswer(*(solver.getUFactor()), rowsU_, colsU_, valsU_, "cpu");

    // Test ILU0 factorization when matrix values change but sparsity is the same
    solver.reset(A);
    status *= verifyAnswer(*(solver.getLFactor()), rowsL_, colsL_, valsL_, "cpu");
    status *= verifyAnswer(*(solver.getUFactor()), rowsU_, colsU_, valsU_, "cpu");

    // Test forward-backward substitution without overwriting rhs
    solver.solve(&rhs, &x);
    status *= verifyAnswer(x, solX_, "cpu");

    // Test forward-backward substitution
    solver.solve(&rhs);
    status *= verifyAnswer(rhs, solX_, "cpu");

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

  // Test matrix:
  //
  //     [     2      0      0      0      1      0      3      0      0]
  //     [     0      7      0      5      0      4      0      0      0]
  //     [     1      0      0      0      3      0      0      2      0]
  //     [     0      0      0      3      0      2      0      0      8]
  // A = [     1      0      0      0      0      0      0      0      0]
  //     [     0      4      0      5      0      1      6      0      0]
  //     [     0      0      2      0      2      0      3      3      0]
  //     [     2      0      0      0      0      0      0      5      1]
  //     [     0      0      7      0      8      0      0      0      4]
  std::vector<index_type> rowsA_ = {0, 3, 6, 9, 12, 13, 17, 21, 24, 27};
  std::vector<index_type> colsA_ = {0, 4, 6,
                                    1, 3, 5,
                                    0, 4, 7,
                                    3, 5, 8,
                                    0,
                                    1, 3, 5, 6,
                                    2, 4, 6, 7,
                                    0, 7, 8,
                                    2, 4, 8};
  std::vector<real_type>  valsA_ = {2., 1., 3., 
                                    7., 5., 4.,
                                    1., 3., 2.,
                                    3., 2., 8.,
                                    1.,
                                    4., 5., 1., 6.,
                                    2., 2., 3., 3.,
                                    2., 5., 1.,
                                    7., 8., 4.};

  /**
   * @brief Create test matrix.
   * 
   * The method creates a block diagonal test matrix from a fixed 9x9
   * sparse blocks.
   * 
   * @todo Currently only single 9x9 sparse matrix is implemented; need to
   * add option to create block diagonal matrix with `k` blocks. 
   * 
   * @param[in] k - multiple of basic matrix pattern (currently unused)
   * @param[in] memspace - string ID of the memory space where matrix is stored
   * 
   */
  matrix::Csr* createCsrMatrix(const index_type /* k */, std::string memspace)
  {

    const index_type N   = static_cast<index_type>(rowsA_.size() - 1);
    const index_type NNZ = static_cast<index_type>(colsA_.size()); 

    // Allocate NxN CSR matrix with NNZ nonzeros
    matrix::Csr* A = new matrix::Csr(N, N, NNZ);
    A->allocateMatrixData(memory::HOST);
    A->updateData(&rowsA_[0], &colsA_[0], &valsA_[0], memory::HOST, memory::HOST);

    // A->print();

    if ((memspace == "cuda") || (memspace == "hip")) {
      A->syncData(memory::DEVICE);
    }

    return A;
  }

  // Lower triangular part of the test matrix A:
  //
  //            [                                                              ]
  //            [     0                                                        ]
  //            [     1      0                                                 ]
  //            [     0      0      0                                          ]
  // lower(L) = [     1      0      0      0                                   ]
  //            [     0      4      0      5      0                            ]
  //            [     0      0      2      0      2      0                     ]
  //            [     2      0      0      0      0      0      0              ]
  //            [     0      0      7      0      8      0      0      0       ]
  std::vector<index_type> rowsAL_ = {0, 0, 0, 1, 1, 2, 4, 6, 7, 9};
  std::vector<index_type> colsAL_ = {0,
                                    0,
                                    1, 3,
                                    2, 4,
                                    0,
                                    2, 4};
  std::vector<real_type>  valsAL_ = {1.0,
                                    1.0,
                                    4.0, 5.0,
                                    2.0, 2.0,
                                    2.0,
                                    7.0, 8.0};

  // Upper triangular part of the test matrix A (zero_diagonal_ = 0.1):
  //
  //            [     2      0      0      0      1      0      3      0      0]
  //            [            7      0      5      0      4      0      0      0]
  //            [                 0.1      0      3      0      0      2      0]
  //            [                          3      0      2      0      0      8]
  // upper(A) = [                               0.1      0      0      0      0]
  //            [                                        1      6      0      0]
  //            [                                               3      3      0]
  //            [                                                      5      1]
  //            [                                                             4]
  std::vector<index_type> rowsAU_ = {0, 3, 6, 9, 12, 13, 15, 17, 19, 20};
  std::vector<index_type> colsAU_ = {0, 4, 6,
                                    1, 3, 5,
                                    2, 4, 7,
                                    3, 5, 8,
                                    4,
                                    5, 6,
                                    6, 7,
                                    7, 8,
                                    8};
  std::vector<real_type>  valsAU_ = {2.0, 1.0, 3.0,
                                    7.0, 5.0, 4.0,
                                    0.1, 3.0, 2.0,
                                    3.0, 2.0, 8.0,
                                    0.1,
                                    1.0, 6.0,
                                    3.0, 3.0,
                                    5.0, 1.0,
                                    4.0};

  // Incomplete factor L of the test matrix (assumes zero_diagonal_ = 0.1):
  //
  //     [                                                              ]
  //     [     0                                                        ]
  //     [   0.5      0                                                 ]
  //     [     0      0      0                                          ]
  // L = [   0.5      0      0      0                                   ]
  //     [     0 0.5714      0 0.7143      0                            ]
  //     [     0      0     20      0    120      0                     ]
  //     [     1      0      0      0      0      0      0              ]
  //     [     0      0     70      0  417.5      0      0      0       ]
  std::vector<index_type> rowsL_ = {0, 0, 0, 1, 1, 2, 4, 6, 7, 9};
  std::vector<index_type> colsL_ = {0,
                                    0,
                                    1, 3,
                                    2, 4,
                                    0,
                                    2, 4};
  std::vector<real_type>  valsL_ = {0.5,
                                    0.5,
                                    5.714285714285714e-01, 7.142857142857144e-01,
                                    20.0, 120.0,
                                    1.0,
                                    70.0, 417.5};

  // Incomplete factor U of the test matrix (assumes zero_diagonal_ = 0.1):
  //
  //     [     2      0      0      0      1      0      3      0      0]
  //     [            7      0      5      0      4      0      0      0]
  //     [                 0.1      0    2.5      0      0      2      0]
  //     [                          3      0      2      0      0      8]
  // U = [                              -0.4      0      0      0      0]
  //     [                                   -2.714      6      0      0]
  //     [                                               3    -37      0]
  //     [                                                      5      1]
  //     [                                                             4]
  std::vector<index_type> rowsU_ = {0, 3, 6, 9, 12, 13, 15, 17, 19, 20};
  std::vector<index_type> colsU_ = {0, 4, 6,
                                    1, 3, 5,
                                    2, 4, 7,
                                    3, 5, 8,
                                    4,
                                    5, 6,
                                    6, 7,
                                    7, 8,
                                    8};
  std::vector<real_type>  valsU_ = {2.0, 1.0, 3.0,
                                    7.0, 5.0, 4.0,
                                    0.1, 2.5, 2.0,
                                    3.0, 2.0, 8.0,
                                   -0.4,
                                   -2.714285714285714, 6.0,
                                    3.0, -37.0,
                                    5.0, 1.0,
                                    4.0};

  /**
   * @brief Compare sparse matrix with a reference.
   * 
   * @param A           - matrix obtained in a test
   * @param answer_rows - reference matrix row data
   * @param answer_cols - reference matrix column data
   * @param answer_vals - reference matrix values
   * @param memspace    - memory space where matrix data is stored
   * @return true  - if elements of the matrix agree with the reference values
   * @return false - otherwise
   * 
   * @todo Only CSR matrices are supported at this time. Need to make this
   * more general.
   */
  bool verifyAnswer(matrix::Sparse& A,
                    const std::vector<index_type>& answer_rows,
                    const std::vector<index_type>& answer_cols,
                    const std::vector<real_type>&  answer_vals,
                    std::string memspace)
  {
    bool status = true;
    if (memspace != "cpu") {
      A.syncData(memory::DEVICE);
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

  /// Reference solution to LUx = 1, where L and U are ILU0 factors
  /// and 1 is vector with all elements set to one.
  std::vector<real_type> solX_ = {-1.889187500000000e+02,
                                  -1.423733082706767e+02,
                                  -2.065000000000000e+02,
                                  -2.461315789473683e+01,
                                  -1.250000000000000e+00,
                                   2.801697368421052e+02,
                                   1.266958333333333e+02,
                                   1.213750000000000e+01,
                                  -6.068750000000000e+01};

  
  /**
   * @brief Compare vector with a reference vector.
   * 
   * @param x        - vector with a result
   * @param answer   - reference solution
   * @param memspace - memory space where the result is stored
   * @return true  - if two vector elements agree to within precision
   * @return false - otherwise
   */
  bool verifyAnswer(vector::Vector& x, const std::vector<real_type>& answer, std::string memspace)
  {
    bool status = true;
    if (memspace != "cpu") {
      x.syncData(memory::DEVICE, memory::HOST);
    }

    for (index_type i = 0; i < x.getSize(); ++i) {
      size_t ii = static_cast<size_t>(i);
      if (!isEqual(x.getData(memory::HOST)[i], answer[ii])) {
        status = false;
        std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                  << ", expected: " << answer[ii] << "\n";
        break; 
      }
    }
    return status;
  }
}; // class MatrixFactorizationTests

}} // namespace ReSolve::tests
