#pragma once

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/LinSolverDirectLUSOL.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <tests/unit/TestBase.hpp>

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
          matrix::Coo* A = createMatrix();

          vector::Vector rhs(A->getNumRows());
          rhs.setToConst(constants::ONE, memory::HOST);

          vector::Vector x(A->getNumColumns());
          x.allocate(memory::HOST);

          if (solver.setup(A) < 0) {
            status *= false;
          }
          if (solver.analyze() < 0) {
            status *= false;
          }
          if (solver.factorize() < 0) {
            status *= false;
          }

          matrix::Csc* L = static_cast<matrix::Csc*>(solver.getLFactor());
          matrix::Csr* U = static_cast<matrix::Csr*>(solver.getUFactor());

          status *= verifyAnswer(*L,
                                 reference_l_factor_rows_,
                                 reference_l_factor_columns_,
                                 reference_l_factor_values_);
          status *= verifyAnswer(*U,
                                 reference_u_factor_rows_,
                                 reference_u_factor_columns_,
                                 reference_u_factor_values_);

          index_type* p_ordering = solver.getPOrdering();

          for (index_type i = 0; i < A->getNumRows(); i++) {
            status *= p_ordering[i] == reference_p_ordering_[i];
          }

          index_type* q_ordering = solver.getQOrdering();

          for (index_type i = 0; i < A->getNumColumns(); i++) {
            status *= q_ordering[i] == reference_q_ordering_[i];
          }

          if (solver.solve(&rhs, &x) < 0) {
            status *= false;
          }

          status *= verifyAnswer(x, solX_);

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
        std::vector<index_type> rowsA_ = {0, 0, 0,
                                          1, 1, 1,
                                          2, 2, 2,
                                          3, 3, 3,
                                          4,
                                          5, 5, 5, 5,
                                          6, 6, 6, 6,
                                          7, 7, 7,
                                          8, 8, 8};
        std::vector<index_type> colsA_ = {0, 4, 6,
                                          1, 3, 5,
                                          0, 4, 7,
                                          3, 5, 8,
                                          0,
                                          1, 3, 5, 6,
                                          2, 4, 6, 7,
                                          0, 7, 8,
                                          2, 4, 8};
        std::vector<real_type> valsA_ = {2., 1., 3.,
                                         7., 5., 4.,
                                         1., 3., 2.,
                                         3., 2., 8.,
                                         1.,
                                         4., 5., 1., 6.,
                                         2., 2., 3., 3.,
                                         2., 5., 1.,
                                         7., 8., 4.};

        /// @brief Creates a test matrix
        matrix::Coo* createMatrix()
        {
          // NOTE: these are hardcoded for now
          index_type size = static_cast<index_type>(valsA_.size());
          matrix::Coo* A = new matrix::Coo(9, 9, size, true, true);
          A->updateData(rowsA_.data(),
                        colsA_.data(),
                        valsA_.data(),
                        memory::HOST,
                        memory::HOST);

          return A;
        }

        /// @brief Reference P ordering
        std::vector<index_type> reference_p_ordering_ = {4, 8, 1, 3, 5, 7, 6, 0, 2};

        /// @brief Reference Q ordering
        std::vector<index_type> reference_q_ordering_ = {0, 2, 1, 3, 5, 7, 8, 6, 4};

        /// @brief Reference lower-triangular L factor rows
        std::vector<index_type> reference_l_factor_rows_ = {5, 7, 8, 0, 6, 1, 4, 2, 4, 3, 4, 6, 8, 5, 8, 6, 8, 7, 8};

        /// @brief Reference lower-triangular L factor columns
        std::vector<index_type> reference_l_factor_columns_ = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8};

        /// @brief Reference lower-triangular L factor values
        std::vector<real_type> reference_l_factor_values_ = {2,
                                                             2,
                                                             1,
                                                             1,
                                                             0.2857142857142857,
                                                             1,
                                                             0.5714285714285714,
                                                             1,
                                                             0.7142857142857144,
                                                             1,
                                                             1,
                                                             0.6000000000000001,
                                                             0.4,
                                                             1,
                                                             0.2295081967213115,
                                                             1,
                                                             -0.2295081967213115,
                                                             1,
                                                             1};

        /// @brief Compare a sparse CSR matrix with a reference specified in COO
        bool verifyAnswer(matrix::Csr& A,
                          const std::vector<index_type>& coo_answer_rows,
                          const std::vector<index_type>& coo_answer_columns,
                          const std::vector<real_type>& coo_answer_values)
        {
          bool status = true;
          index_type i = 0;

          index_type* rows = A.getRowData(memory::HOST);
          index_type* columns = A.getColData(memory::HOST);
          real_type* values = A.getValues(memory::HOST);

          for (index_type row = 0; row < A.getNumRows(); row++) {
            for (index_type offset = rows[row]; offset < rows[row + 1]; offset++) {
              if (row != coo_answer_rows[i]
                  || columns[offset] != coo_answer_columns[i]
                  || !isEqual(values[offset], coo_answer_values[i])) {
                std::cout << std::setprecision(16)
                          << "i = " << i << ", ("
                          << coo_answer_rows[i] << ", "
                          << coo_answer_columns[i] << ", "
                          << coo_answer_values[i] << ") != ("
                          << row << ", "
                          << columns[offset] << ", "
                          << values[offset] << ")\n";
                status = false;
              }
              i++;
            }
          }

          return status;
        }

        /// @brief Reference upper triangular U factor rows
        std::vector<index_type> reference_u_factor_rows_ = {0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8};

        /// @brief Reference upper triangular U factor columns
        std::vector<index_type> reference_u_factor_columns_ = {0, 1, 6, 8, 2, 3, 4, 3, 4, 6, 4, 6, 7, 5, 6, 6, 7, 8, 7, 8, 8};

        /// @brief Reference upper triangular U factor values
        std::vector<real_type> reference_u_factor_values_ = {1,
                                                             7,
                                                             4,
                                                             8,
                                                             7,
                                                             5,
                                                             4,
                                                             3,
                                                             2,
                                                             8,
                                                             -2.714285714285714,
                                                             -5.714285714285715,
                                                             6,
                                                             5,
                                                             1,
                                                             -1.742857142857143,
                                                             3,
                                                             -0.2857142857142856,
                                                             3,
                                                             1,
                                                             3.295081967213115};

        /// @brief Compare a sparse CSC matrix with a reference specified in COO
        bool verifyAnswer(matrix::Csc& A,
                          const std::vector<index_type>& coo_answer_rows,
                          const std::vector<index_type>& coo_answer_columns,
                          const std::vector<real_type>& coo_answer_values)
        {
          bool status = true;
          index_type i = 0;

          index_type* columns = A.getColData(memory::HOST);
          index_type* rows = A.getRowData(memory::HOST);
          real_type* values = A.getValues(memory::HOST);

          for (index_type column = 0; column < A.getNumColumns(); column++) {
            for (index_type offset = columns[column];
                 offset < columns[column + 1];
                 offset++) {
              if (column != coo_answer_columns[i]
                  || rows[offset] != coo_answer_rows[i]
                  || !isEqual(values[offset], coo_answer_values[i])) {
                std::cout << std::setprecision(16)
                          << "i = " << i << ", ("
                          << coo_answer_rows[i] << ", "
                          << coo_answer_columns[i] << ", "
                          << coo_answer_values[i] << ") != ("
                          << rows[offset] << ", "
                          << column << ", "
                          << values[offset] << ")\n";
                status = false;
              }
              i++;
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
                          const std::vector<real_type>& answer)
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
