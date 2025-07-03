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

      TestOutcome ruizTest()
      {
        index_type      n = 1024;
        matrix::Csr*    A = new matrix::Csr(n, n, 2 * n - 1);
        matrix::Csr*    H = new matrix::Csr(n, n, n);
        vector::Vector* rhs_top = new vector::Vector(n);
        vector::Vector* rhs_bottom = new vector::Vector(n);
        generateMatrixData(&A, &H, &rhs_top, &rhs_bottom, n);

        // Transpose A and store in A_tr
        matrix::Csr* A_tr = new matrix::Csr(n, n, 2 * n - 1);
        A_tr->allocateMatrixData(memspace_);
        matrixHandler_.transpose(A, A_tr, memspace_);

        // Perform scaling
        index_type                 num_iterations = 2;
        index_type                 total_n        = 2 * n;
        ReSolve::hykkt::RuizScaling RuizScaling(n, total_n, memspace_);
        RuizScaling.addMatrixData(H, A, A_tr);
        RuizScaling.addRhsData(rhs_top, rhs_bottom);
        RuizScaling.scale(num_iterations);
        vector::Vector* aggregate_scaling_vector = RuizScaling.getAggregateScalingVector();
        // Get data back to HOST
        if (memspace_ == memory::DEVICE)
        {
          A->syncData(memory::HOST);
          H->syncData(memory::HOST);
          A_tr->syncData(memory::HOST);
          rhs_top->syncData(memory::HOST);
          rhs_bottom->syncData(memory::HOST);
          aggregate_scaling_vector->syncData(memory::HOST);
        }

        bool         test_passed = true;
        const double tol         = 1e-8;
        if (fabs(H->getValues(memory::HOST)[n / 2 - 1] - 0.062378167641326) > tol)
        {
          test_passed = false;
          std::cout << "Test failed: H[n/2-1][n/2-1] = " << H->getValues(memory::HOST)[n / 2 - 1]
                    << ", expected " << 0.062378167641326 << "\n";
        }

        if (fabs(A->getValues(memory::HOST)[A->getNnz() - 1] - 0.005524271728020) > tol)
        {
          test_passed = false;
          std::cout << "Test failed: A[n-1][n-1] = " << A->getValues(memory::HOST)[A->getNnz() - 1]
                    << ", expected " << 0.005524271728020 << "\n";
        }

        if (fabs(A_tr->getValues(memory::HOST)[1] - 0.5) > tol)
        {
          test_passed = false;
          std::cout << "Test failed: A_tr[0][1] = " << A_tr->getValues(memory::HOST)[1]
                    << ", expected " << 0.5 << "\n";
        }

        if (fabs(rhs_top->getData(memory::HOST)[n / 2 - 1] - 0.044151078568835) > tol)
        {
          test_passed = false;
          std::cout << "Test failed: rhs[n/2 - 1] = " << rhs_top->getData(memory::HOST)[n / 2 - 1]
                    << ", expected " << 0.044151078568835 << "\n";
        }

        if (fabs(rhs_bottom->getData(memory::HOST)[n / 2 - 1] - 0.044194173824159) > tol)
        {
          test_passed = false;
          std::cout << "Test failed: rhs[3*n/2 - 1] = " << rhs_bottom->getData(memory::HOST)[n / 2 - 1]
                    << ", expected " << 0.044194173824159 << "\n";
        }

        if (fabs(aggregate_scaling_vector->getData(memory::HOST)[32] - 0.171498585142) > tol)
        {
          test_passed = false;
          std::cout << "Test failed: aggregate_scaling_vector[32] = " << aggregate_scaling_vector->getData(memory::HOST)[32]
                    << ", expected " << 0.171498585142 << "\n";
        }

        if (test_passed)
        {
          std::cout << "Test passed successfully.\n";
        }

        delete A;
        delete H;
        delete A_tr;
        delete rhs_top;
        delete rhs_bottom;

        return test_passed ? PASS : FAIL;
      }

    private:
      memory::MemorySpace memspace_;
      MemoryHandler       mem_;
      MatrixHandler&      matrixHandler_;

      /**
       * @brief Generate matrix data for testing
       *
       * Creates n x n matrices A and H where A has ones on the diagonal and
       * 2, 3, ..., n below the diagonal, and H has sqrt(n) on the diagonal.
       *
       * @param[out] A Pointer to the matrix A (CSR format)
       * @param[out] H Pointer to the matrix H (CSR format)
       * @param[out] rhs Pointer to the right-hand side vector
       * @param[in] n Size of the matrices and vectors
       */
      void generateMatrixData(matrix::Csr** A, matrix::Csr** H, vector::Vector** rhs_top, vector::Vector** rhs_bottom, index_type n)
      {
        // Define A
        index_type* A_row_data = new index_type[n + 1];
        index_type* A_col_data = new index_type[2 * n - 1];
        real_type*  A_val_data = new real_type[2 * n - 1];
        A_row_data[0]          = 0;
        for (index_type i = 0; i < n; ++i)
        {
          if (i > 0)
          {
            A_val_data[i * 2 - 1] = i + 1.0;
            A_col_data[i * 2 - 1] = i - 1;
            A_row_data[i]         = i * 2 - 1;
          }
          A_val_data[i * 2] = 1.0;
          A_col_data[i * 2] = i;
        }
        A_row_data[n] = 2 * n - 1;

        (*A)->copyDataFrom(A_row_data, A_col_data, A_val_data, memory::HOST, memspace_);

        // Define H
        index_type* H_row_data = new index_type[n + 1];
        index_type* H_col_data = new index_type[n];
        real_type*  H_val_data = new real_type[n];
        for (index_type i = 0; i < n; ++i)
        {
          H_row_data[i] = i;
          H_col_data[i] = i;
          H_val_data[i] = sqrt(n);
        }
        H_row_data[n] = n;

        (*H)->copyDataFrom(H_row_data, H_col_data, H_val_data, memory::HOST, memspace_);

        // Define rhs vectors
        real_type* rhs_data = new real_type[n];
        for (index_type i = 0; i < n; ++i)
        {
          rhs_data[i] = 1.0;
        }

        (*rhs_top)->copyDataFrom(rhs_data, memory::HOST, memspace_);

        (*rhs_bottom)->copyDataFrom(rhs_data, memory::HOST, memspace_);

        delete[] A_row_data;
        delete[] A_col_data;
        delete[] A_val_data;
        delete[] H_row_data;
        delete[] H_col_data;
        delete[] H_val_data;
        delete[] rhs_data;
      }
    }; // class HykktPermutationTests
  } // namespace tests
} // namespace ReSolve
