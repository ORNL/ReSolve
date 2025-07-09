/**
 * @file HykktRuizScalingTests.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Implementation of tests for class hykkt::RuizScaling
 *
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/MemoryUtils.hpp>
#include <resolve/hykkt/ruiz/RuizScaling.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Tests for class hykkt::RuizScaling
     *
     */
    class HykktRuizScalingTests : public TestBase
    {
    public:
      HykktRuizScalingTests(memory::MemorySpace memspace, MatrixHandler& matrixHandler)
        : memspace_(memspace), matrixHandler_(matrixHandler)
      {
      }

      virtual ~HykktRuizScalingTests()
      {
      }

      /**
       * @brief Test the RuizScaling implementation
       *
       * Generates nontrivial Hessian of size n x n and Jacobian of size (n-1) x n,
       * where n is the input parameter. See generateMatrixData for details.
       *
       * Performs two iterations of Ruiz scaling on the system and verifies the result
       * against the symbolic expected values.
       *
       * @param n
       * @return TestOutcome Result of the test
       */
      TestOutcome ruizTest(index_type n)
      {
        matrix::Csr* H = new matrix::Csr(n, n, 3 * n - 2);
        matrix::Csr* J = new matrix::Csr(n - 1, n, 2 * n - 2);

        vector::Vector* rhs_top    = new vector::Vector(n);
        vector::Vector* rhs_bottom = new vector::Vector(n - 1);
        generateMatrixData(H, J, rhs_top, rhs_bottom, n);

        // // Transpose J and store in J_tr
        matrix::Csr* J_tr = new matrix::Csr(n, n - 1, 2 * n - 2);
        J_tr->allocateMatrixData(memspace_);
        matrixHandler_.transpose(J, J_tr, memspace_);

        // Perform scaling
        index_type                  num_iterations = 2;
        index_type                  total_n        = 2 * n - 1;
        ReSolve::hykkt::RuizScaling RuizScaling(n, total_n, memspace_);
        RuizScaling.addMatrixData(H, J, J_tr);
        RuizScaling.addRhsData(rhs_top, rhs_bottom);
        RuizScaling.scale(num_iterations);
        vector::Vector* aggregate_scaling_vector = RuizScaling.getAggregateScalingVector();
        // Get data back to HOST
        if (memspace_ == memory::DEVICE)
        {
          H->syncData(memory::HOST);
          J->syncData(memory::HOST);
          J_tr->syncData(memory::HOST);
          rhs_top->syncData(memory::HOST);
          rhs_bottom->syncData(memory::HOST);
          aggregate_scaling_vector->syncData(memory::HOST);
        }

        TestStatus  status;
        std::string testname(__func__);
        testname += " n=" + std::to_string(n);
        status *= validateResult(H, J, rhs_top, rhs_bottom, aggregate_scaling_vector, n);

        delete H;
        delete J;
        delete J_tr;
        delete rhs_top;
        delete rhs_bottom;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;
      MatrixHandler&      matrixHandler_;

      /**
       * @brief Generate matrix data for testing
       *
       * Creates Hessian matrix H of size n x n with 1, 2, ..., n on the diagonal and 1 on the diagonals
       * above and below the main diagonal. The Jacobian matrix J of size (n-1) x n is created with 1 on the
       * diagonal and 1 on the superdiagonal.
       *
       * @param[in,out] H Pointer to the matrix H (CSR format)
       * @param[in,out] J Pointer to the matrix J (CSR format)
       * @param[in,out] rhs_top Pointer to the top part of the right-hand side vector
       * @param[in,out] rhs_bottom Pointer to the bottom part of the right-hand side vector
       * @param[in] n Size parameter
       */
      void generateMatrixData(matrix::Csr* H, matrix::Csr* J, vector::Vector* rhs_top, vector::Vector* rhs_bottom, index_type n)
      {
        // Define H
        index_type* H_row_data = new index_type[n + 1];
        index_type* H_col_data = new index_type[3 * n - 2];
        real_type*  H_val_data = new real_type[3 * n - 2];

        H_row_data[0] = 0;
        H_val_data[0] = 1.0;
        H_col_data[0] = 0;
        H_val_data[1] = 1.0;
        H_col_data[1] = 1;

        H_row_data[1] = 2;
        for (index_type i = 2; i < n; ++i)
        {
          index_type start_row_index = H_row_data[i - 1];
          H_row_data[i]              = start_row_index + 3;

          H_val_data[start_row_index] = 1.0;
          H_col_data[start_row_index] = i - 2;

          H_val_data[start_row_index + 1] = i;
          H_col_data[start_row_index + 1] = i - 1;

          H_val_data[start_row_index + 2] = 1.0;
          H_col_data[start_row_index + 2] = i;
        }
        H_row_data[n]         = 3 * n - 2;
        H_val_data[3 * n - 4] = 1.0;
        H_col_data[3 * n - 4] = n - 2;
        H_val_data[3 * n - 3] = n;
        H_col_data[3 * n - 3] = n - 1;

        H->copyDataFrom(H_row_data, H_col_data, H_val_data, memory::HOST, memspace_);

        // Define J
        index_type* J_row_data = new index_type[n];
        index_type* J_col_data = new index_type[2 * n - 2];
        real_type*  J_val_data = new real_type[2 * n - 2];
        J_row_data[0]          = 0;
        for (index_type i = 0; i < n - 1; ++i)
        {
          J_row_data[i + 1] = 2 * (i + 1);

          J_val_data[2 * i] = 1.0;
          J_col_data[2 * i] = i;

          J_val_data[2 * i + 1] = 1.0;
          J_col_data[2 * i + 1] = i + 1;
        }
        J_row_data[n - 1] = 2 * n - 2;

        J->copyDataFrom(J_row_data, J_col_data, J_val_data, memory::HOST, memspace_);

        // Define rhs vectors
        real_type* rhs_data = new real_type[n];
        for (index_type i = 0; i < n; ++i)
        {
          rhs_data[i] = 1.0;
        }

        rhs_top->copyDataFrom(rhs_data, memory::HOST, memspace_);

        rhs_bottom->copyDataFrom(rhs_data, memory::HOST, memspace_);

        delete[] H_row_data;
        delete[] H_col_data;
        delete[] H_val_data;

        delete[] J_row_data;
        delete[] J_col_data;
        delete[] J_val_data;

        delete[] rhs_data;
      }

      /**
       * @brief Validate the results of the scaling
       * @param[in] H Pointer to the scaled Hessian matrix
       * @param[in] J Pointer to the scaled Jacobian matrix
       * @param[in] rhs_top Pointer to the top part of the right-hand side vector
       * @param[in] rhs_bottom Pointer to the bottom part of the right-hand side vector
       * @param[in] aggregate_scaling_vector Pointer to the aggregate scaling vector
       * @param[in] n Size of the matrices and vectors
       */
      bool validateResult(matrix::Csr* H, matrix::Csr* J, vector::Vector* rhs_top, vector::Vector* rhs_bottom, vector::Vector* aggregate_scaling_vector, index_type n)
      {
        bool      test_passed = true;
        real_type tol         = 1e-8;

        // Check the diagonal of H
        // H[i][i] = 1
        for (index_type i = 0; i < n; ++i)
        {
          real_type expected_value = 1.0;
          if (std::abs(H->getValues(memory::HOST)[i * 3] - expected_value) > tol)
          {
            std::cout << "Test failed for H[" << i << "," << i << "]: expected " << expected_value
                      << ", got " << H->getValues(memory::HOST)[i * 3] << "\n";
            test_passed = false;
          }
        }

        // Check the off-diagonals of H
        // H[i][j] = 1 / sqrt((i+1) * (j+1)) if |i - j| = 1
        index_type row = 0;
        index_type col = 1;
        for (index_type i = 1; i < H->getNnz(); i += 3)
        {
          real_type expected_value = 1.0 / std::sqrt(static_cast<real_type>((row + 1) * (col + 1)));
          if (std::abs(H->getValues(memory::HOST)[i] - expected_value) > tol)
          {
            std::cout << "Test failed for H[" << row << "," << col << "]: expected " << expected_value
                      << ", got " << H->getValues(memory::HOST)[i] << "\n";
            test_passed = false;
          }
          // The next entry is the element symmetric across the diagonal
          if (std::abs(H->getValues(memory::HOST)[i + 1] - expected_value) > tol)
          {
            std::cout << "Test failed for H[" << col << "," << row << "]: expected " << expected_value
                      << ", got " << H->getValues(memory::HOST)[i + 1] << "\n";
            test_passed = false;
          }
          row++;
          col++;
        }

        // Check J
        // J[i][j] = (i+1)^{0.25} / (j+1)^{0.5} for i = j or j = i+1
        row = 0;
        col = 0;
        for (index_type i = 0; i < J->getNnz(); ++i)
        {
          real_type expected_value = std::pow(static_cast<real_type>(row + 1), 0.25) / std::pow(static_cast<real_type>(col + 1), 0.5);
          if (std::abs(J->getValues(memory::HOST)[i] - expected_value) > tol)
          {
            std::cout << "Test failed for J[" << row << "," << col << "]: expected " << expected_value
                      << ", got " << J->getValues(memory::HOST)[i] << "\n";
            test_passed = false;
          }

          if (i % 2 == 0)
          {
            col++;
          }
          else
          {
            row++;
          }
        }

        // Check rhs_top
        // rhs[i] = 1 / sqrt(i + 1) for i = 0, 1, ..., n-1
        for (index_type i = 0; i < n; ++i)
        {
          real_type expected_value = 1.0 / std::sqrt(static_cast<real_type>(i + 1));
          if (std::abs(rhs_top->getData(memory::HOST)[i] - expected_value) > tol)
          {
            std::cout << "Test failed for rhs_top[" << i << "]: expected " << expected_value
                      << ", got " << rhs_top->getData(memory::HOST)[i] << "\n";
            test_passed = false;
          }
        }

        // Check rhs_bottom
        // rhs[i + n] = (i + 1)^{0.25} for i = 0, 1, ..., n-2
        for (index_type i = 0; i < n - 1; ++i)
        {
          real_type expected_value = std::pow(static_cast<real_type>(i + 1), 0.25);
          if (std::abs(rhs_bottom->getData(memory::HOST)[i] - expected_value) > tol)
          {
            std::cout << "Test failed for rhs_bottom[" << i << "]: expected " << expected_value
                      << ", got " << rhs_bottom->getData(memory::HOST)[i] << "\n";
            test_passed = false;
          }
        }

        // Check aggregate scaling vector top
        // aggregate_scaling_vector[i] = 1 / sqrt(i + 1) for i = 0, 1, ..., n-1
        for (index_type i = 0; i < n; ++i)
        {
          real_type expected_value = 1.0 / std::sqrt(static_cast<real_type>(i + 1));
          if (std::abs(aggregate_scaling_vector->getData(memory::HOST)[i] - expected_value) > tol)
          {
            std::cout << "Test failed for aggregate_scaling_vector[" << i << "]: expected " << expected_value
                      << ", got " << aggregate_scaling_vector->getData(memory::HOST)[i] << "\n";
            test_passed = false;
          }
        }

        // Check aggregate scaling vector bottom
        // aggregate_scaling_vector[i + n] = (i + 1)^{0.25} for i = 0, 1, ..., n-2
        for (index_type i = 0; i < n - 1; ++i)
        {
          real_type expected_value = std::pow(static_cast<real_type>(i + 1), 0.25);
          if (std::abs(aggregate_scaling_vector->getData(memory::HOST)[i + n] - expected_value) > tol)
          {
            std::cout << "Test failed for aggregate_scaling_vector[" << i + n << "]: expected " << expected_value
                      << ", got " << aggregate_scaling_vector->getData(memory::HOST)[i + n] << "\n";
            test_passed = false;
          }
        }

        return test_passed;
      }
    }; // class HykktPermutationTests
  } // namespace tests
} // namespace ReSolve
