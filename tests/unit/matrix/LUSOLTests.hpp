#pragma once

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <tests/unit/TestBase.hpp>

#include <resolve/LinSolverDirectLUSOL.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve
{
  namespace tests
  {

    /**
     * @class Unit tests for LUSOL-related operations
     */
    class LUSOLTests : TestBase
    {
      public:
        LUSOLTests()
        {
        }
        virtual ~LUSOLTests()
        {
        }

        TestOutcome lusolConstructor()
        {
          TestStatus status;
          status.skipTest();

          return status.report(__func__);
        }

        TestOutcome simpleSolve()
        {
          TestStatus status;

          LinSolverDirectLUSOL solver;
          matrix::Csr* A = createCsrMatrix(0, "cpu");

          vector::Vector rhs(A->getNumRows());
          rhs.setToConst(constants::ONE, memory::HOST);

          vector::Vector x(A->getNumColumns());
          x.allocate(memory::HOST);

          solver.setup(A);
          if (solver.factorize() < 0) {
            status *= false;
          }
          solver.solve(&rhs, &x);

          status *= verifyAnswer(x, solX_, "cpu");

          delete A;

          return status.report(__func__);
        }

      private:
        ReSolve::MatrixHandler* createMatrixHandler()
        {
          LinAlgWorkspaceCpu* workspace = new LinAlgWorkspaceCpu();
          return new MatrixHandler(workspace);
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
        std::vector<index_type> colsA_ = {0, 4, 6, 1, 3, 5, 0, 4, 7,
                                          3, 5, 8, 0, 1, 3, 5, 6, 2,
                                          4, 6, 7, 0, 7, 8, 2, 4, 8};
        std::vector<real_type> valsA_ = {2., 1., 3., 7., 5., 4., 1., 3., 2.,
                                         3., 2., 8., 1., 4., 5., 1., 6., 2.,
                                         2., 3., 3., 2., 5., 1., 7., 8., 4.};

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
         * @param[in] memspace - string ID of the memory space where matrix is
         * stored
         *
         */
        matrix::Csr* createCsrMatrix(const index_type /* k */,
                                     std::string memspace)
        {

          const index_type N = static_cast<index_type>(rowsA_.size() - 1);
          const index_type NNZ = static_cast<index_type>(colsA_.size());

          // Allocate NxN CSR matrix with NNZ nonzeros
          matrix::Csr* A = new matrix::Csr(N, N, NNZ);
          A->allocateMatrixData(memory::HOST);
          A->updateData(&rowsA_[0],
                        &colsA_[0],
                        &valsA_[0],
                        memory::HOST,
                        memory::HOST);

          // A->print();

          if ((memspace == "cuda") || (memspace == "hip")) {
            A->copyData(memory::DEVICE);
          }

          return A;
        }

        /**
         * @brief Compare sparse matrix with a reference.
         *
         * @param A           - matrix obtained in a test
         * @param answer_rows - reference matrix row data
         * @param answer_cols - reference matrix column data
         * @param answer_vals - reference matrix values
         * @param memspace    - memory space where matrix data is stored
         * @return true  - if elements of the matrix agree with the reference
         * values
         * @return false - otherwise
         *
         * @todo Only CSR matrices are supported at this time. Need to make this
         * more general.
         */
        bool verifyAnswer(matrix::Sparse& A,
                          const std::vector<index_type>& answer_rows,
                          const std::vector<index_type>& answer_cols,
                          const std::vector<real_type>& answer_vals,
                          std::string memspace)
        {
          bool status = true;

          size_t N = static_cast<size_t>(A.getNumRows());
          for (size_t i = 0; i <= N; ++i) {
            if (A.getRowData(memory::HOST)[i] != answer_rows[i]) {
              status = false;
              std::cout << "Matrix row pointer rows[" << i
                        << "] = " << A.getRowData(memory::HOST)[i]
                        << ", expected: " << answer_rows[i] << "\n";
            }
          }

          size_t NNZ = static_cast<size_t>(A.getNnz());
          for (size_t i = 0; i < NNZ; ++i) {
            if (A.getColData(memory::HOST)[i] != answer_cols[i]) {
              status = false;
              std::cout << "Matrix column index cols[" << i
                        << "] = " << A.getColData(memory::HOST)[i]
                        << ", expected: " << answer_cols[i] << "\n";
            }
            if (!isEqual(A.getValues(memory::HOST)[i], answer_vals[i])) {
              status = false;
              std::cout << "Matrix value element vals[" << i
                        << "] = " << A.getValues(memory::HOST)[i]
                        << ", expected: " << answer_vals[i] << "\n";
              // break;
            }
          }
          return status;
        }

        /// @brief Reference solution to Ax = rhs
        std::vector<real_type> solX_ = {1,
                                        -2.7715806930261,
                                        0.930348258706468,
                                        2.37505455180239,
                                        -0.0398009950248756,
                                        2.13144802304268,
                                        -0.320066334991708,
                                        0.0597014925373134,
                                        -1.29850746268657};

        /**
         * @brief Compare vector with a reference vector.
         *
         * @param x        - vector with a result
         * @param answer   - reference solution
         * @param memspace - memory space where the result is stored
         * @return true  - if two vector elements agree to within precision
         * @return false - otherwise
         */
        bool verifyAnswer(vector::Vector& x,
                          const std::vector<real_type>& answer,
                          std::string memspace)
        {
          bool status = true;

          for (index_type i = 0; i < x.getSize(); ++i) {
            size_t ii = static_cast<size_t>(i);
            if (!isEqual(x.getData(memory::HOST)[i], answer[ii])) {
              status = false;
              std::cout << "Solution vector element x[" << i
                        << "] = " << x.getData(memory::HOST)[i]
                        << ", expected: " << answer[ii] << "\n";
              break;
            }
          }

          return status;
        }
    }; // class MatrixFactorizationTests
  }    // namespace tests
} // namespace ReSolve
