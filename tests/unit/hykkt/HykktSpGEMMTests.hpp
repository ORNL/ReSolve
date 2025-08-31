
#pragma once

#include <algorithm>
#include <cholmod.h>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/hykkt/spgemm/SpGEMM.hpp>
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
     * @brief Tests for class hykkt::SpGEMM
     *
     */
    class HykktSpGEMMTests : public TestBase
    {
    public:
      HykktSpGEMMTests(memory::MemorySpace memspace)
        : memspace_(memspace)
      {
      }

      virtual ~HykktSpGEMMTests()
      {
      }

      /**
       * @brief Test the solver on a minimal example
       *
       * @return TestOutcome the outcome of the test
       */
      TestOutcome minimalCorrectness()
      {
        TestStatus  status;
        std::string testname(__func__);

        hykkt::SpGEMM spgemm(memspace_, 1.0, 1.0);

        matrix::Csr* A = nullptr;
        matrix::Csr* B = nullptr;
        matrix::Csr* D = nullptr;

        generateExampleData(&A, &B, &D);

        matrix::Csr* E = nullptr;

        spgemm.loadProductMatrices(A, B);
        spgemm.loadSumMatrix(D);
        spgemm.loadResultMatrix(&E);
        spgemm.compute();

        if (memspace_ == memory::DEVICE)
        {
          E->syncData(memory::HOST);
        }

        status *= verifyResult(E, 1.0);

        delete A;
        delete B;
        delete D;
        delete E;

        return status.report(testname.c_str());
      }

      /**
       * @brief Generate an n by n example with known solution
       *
       * Generates lower bidiagonal A, upper bidiagonal B, and D with
       * ones on upper diagonal.
       *
       * @param[in] n - The size of the test
       */
      TestOutcome symbolic(index_type n)
      {
        TestStatus  status;
        std::string testname(__func__);
        testname += ", n = " + std::to_string(n);

        hykkt::SpGEMM spgemm(memspace_, 1.0, 1.0);

        index_type nnz = 2 * n - 1;

        index_type* A_row_ptr = new index_type[n + 1];
        index_type* A_col_ind = new index_type[nnz];
        real_type*  A_values  = new real_type[nnz];

        index_type* B_row_ptr = new index_type[n + 1];
        index_type* B_col_ind = new index_type[nnz];
        real_type*  B_values  = new real_type[nnz];

        index_type* D_row_ptr = new index_type[n + 1];
        index_type* D_col_ind = new index_type[n - 1];
        real_type*  D_values  = new real_type[n - 1];

        A_row_ptr[0] = 0;
        A_row_ptr[1] = 1;
        A_col_ind[0] = 0;
        A_values[0]  = 1.0;

        // A = [1      ]
        //     [2 1    ]
        //     [  3 1  ]
        //     [  ...  ]
        for (index_type i = 1; i < n; i++)
        {
          A_col_ind[2 * i - 1] = i - 1;
          A_values[2 * i - 1]  = i + 1;

          A_col_ind[2 * i] = i;
          A_values[2 * i]  = 1.0;

          A_row_ptr[i + 1] = 2 + A_row_ptr[i];
        }

        // B = [1 2    ]
        //     [  1 3  ]
        //     [    1  ]
        //     [  ...  ]
        B_row_ptr[0] = 0;
        for (index_type i = 0; i < n; i++)
        {
          B_col_ind[2 * i] = i;
          B_values[2 * i]  = 1.0;

          if (i < n - 1)
          {
            B_col_ind[2 * i + 1] = i + 1;
            B_values[2 * i + 1]  = i + 2;
            B_row_ptr[i + 1]     = 2 + B_row_ptr[i];
          }
          else
          {
            B_row_ptr[i + 1] = 1 + B_row_ptr[i];
          }
        }

        // D = [0 1    ]
        //     [  0 1  ]
        //     [    0 1]
        //     [  ...  ]
        D_row_ptr[0] = 0;
        for (index_type i = 0; i < n - 1; i++)
        {
          D_col_ind[i]     = i + 1;
          D_values[i]      = 1.0;
          D_row_ptr[i + 1] = 1 + D_row_ptr[i];
        }
        D_row_ptr[n] = D_row_ptr[n - 1];

        matrix::Csr* A = new matrix::Csr(n, n, nnz);
        matrix::Csr* B = new matrix::Csr(n, n, nnz);
        matrix::Csr* D = new matrix::Csr(n, n, n - 1);

        A->copyDataFrom(A_row_ptr, A_col_ind, A_values, memory::HOST, memspace_);
        B->copyDataFrom(B_row_ptr, B_col_ind, B_values, memory::HOST, memspace_);
        D->copyDataFrom(D_row_ptr, D_col_ind, D_values, memory::HOST, memspace_);

        matrix::Csr* E = nullptr;

        spgemm.loadProductMatrices(A, B);
        spgemm.loadSumMatrix(D);
        spgemm.loadResultMatrix(&E);
        spgemm.compute();

        if (memspace_ == memory::DEVICE)
        {
          E->syncData(memory::HOST);
        }

        double tol = 1e-6;
        if (fabs(E->getValues(memory::HOST)[0] - 1.0) > tol)
        {
          std::cerr << "Test failed: E[0][0] = " << E->getValues(memory::HOST)[0] << ", expected: 1.0\n";
          status *= false;
        }

        if (fabs(E->getValues(memory::HOST)[1] - 3.0) > tol)
        {
          std::cerr << "Test failed: E[0][1] = " << E->getValues(memory::HOST)[1] << ", expected: 2.0\n";
          status *= false;
        }

        for (index_type i = 1; i < n; i++)
        {
          if (fabs(E->getValues(memory::HOST)[3 * i - 1] - (i + 1)) > tol)
          {
            std::cerr << "Test failed: E[" << i << "][" << i - 1 << "] = " << E->getValues(memory::HOST)[3 * i - 1] << ", expected: " << (i + 1) << "\n";
            status *= false;
          }

          if (fabs(E->getValues(memory::HOST)[3 * i] - (1 + (i + 1) * (i + 1))) > tol)
          {
            std::cerr << "Test failed: E[" << i << "][" << i << "] = " << E->getValues(memory::HOST)[3 * i] << ", expected: " << (1 + (i + 1) * (i + 1)) << "\n";
            status *= false;
          }

          if (i == n - 1)
          {
            break;
          }

          if (fabs(E->getValues(memory::HOST)[3 * i + 1] - (i + 3)) > tol)
          {
            std::cerr << "Test failed: E[" << i << "][" << i + 1 << "] = " << E->getValues(memory::HOST)[3 * i + 1] << ", expected: " << (i + 3) << "\n";
            status *= false;
          }
        }

        delete[] A_row_ptr;
        delete[] A_col_ind;
        delete[] A_values;

        delete[] B_row_ptr;
        delete[] B_col_ind;
        delete[] B_values;

        delete[] D_row_ptr;
        delete[] D_col_ind;
        delete[] D_values;

        delete A;
        delete B;
        delete D;
        delete E;

        return status.report(testname.c_str());
      }

      TestOutcome reuse()
      {
        TestStatus  status;
        std::string testname(__func__);

        hykkt::SpGEMM spgemm(memspace_, 2.0, 2.0);

        matrix::Csr* A = nullptr;
        matrix::Csr* B = nullptr;
        matrix::Csr* D = nullptr;

        generateExampleData(&A, &B, &D);

        matrix::Csr* E = nullptr;

        spgemm.loadProductMatrices(A, B);
        spgemm.loadSumMatrix(D);
        spgemm.loadResultMatrix(&E);
        spgemm.compute();

        if (memspace_ == memory::DEVICE)
        {
          E->syncData(memory::HOST);

          A->syncData(memory::HOST);
          D->syncData(memory::HOST);
        }

        status *= verifyResult(E, 2.0);

        for (index_type j = 0; j < A->getNnz(); j++)
        {
          A->getValues(memory::HOST)[j] *= 2.0;
        }
        for (index_type j = 0; j < D->getNnz(); j++)
        {
          D->getValues(memory::HOST)[j] *= 2.0;
        }
        A->setUpdated(memory::HOST);
        D->setUpdated(memory::HOST);

        if (memspace_ == memory::DEVICE)
        {
          A->syncData(memory::DEVICE);
          D->syncData(memory::DEVICE);
        }

        spgemm.loadProductMatrices(A, B);
        spgemm.loadSumMatrix(D);
        spgemm.compute();

        if (memspace_ == memory::DEVICE)
        {
          E->syncData(memory::HOST);
        }

        status *= verifyResult(E, 4.0);

        delete A;
        delete B;
        delete D;
        delete E;

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;

      void generateExampleData(matrix::Csr** A, matrix::Csr** B, matrix::Csr** D)
      {
        index_type* A_row_ptr = new index_type[4]{0, 1, 3, 5};
        index_type* A_col_ind = new index_type[5]{0, 1, 2, 0, 2};
        real_type*  A_values  = new real_type[5]{1.0, 2.0, -4.0, 5.0, 3.0};

        index_type* B_row_ptr = new index_type[4]{0, 1, 3, 5};
        index_type* B_col_ind = new index_type[5]{0, 0, 1, 1, 2};
        real_type*  B_values  = new real_type[5]{-1.0, -2.0, 3.0, 2.0, 4.0};

        index_type* D_row_ptr = new index_type[4]{0, 2, 4, 6};
        index_type* D_col_ind = new index_type[6]{0, 2, 0, 1, 1, 2};
        real_type*  D_values  = new real_type[6]{3.0, -4.0, -2.0, 1.0, 5.0, 7.0};

        *A = new matrix::Csr(3, 3, 5);
        *B = new matrix::Csr(3, 3, 5);
        *D = new matrix::Csr(3, 3, 6);

        (*A)->copyDataFrom(A_row_ptr, A_col_ind, A_values, memory::HOST, memspace_);
        (*B)->copyDataFrom(B_row_ptr, B_col_ind, B_values, memory::HOST, memspace_);
        (*D)->copyDataFrom(D_row_ptr, D_col_ind, D_values, memory::HOST, memspace_);

        delete[] A_row_ptr;
        delete[] A_col_ind;
        delete[] A_values;

        delete[] B_row_ptr;
        delete[] B_col_ind;
        delete[] B_values;

        delete[] D_row_ptr;
        delete[] D_col_ind;
        delete[] D_values;
      }

      bool verifyResult(matrix::Csr* E, real_type multiplier)
      {
        bool is_correct = true;

        index_type* E_row_ptr_expected = new index_type[4]{0, 2, 5, 8};
        index_type* E_col_ind_expected = new index_type[8]{0, 2, 0, 1, 2, 0, 1, 2};
        real_type*  E_values_expected  = new real_type[8]{2.0, -4.0, -6.0, -1.0, -16, -5.0, 11, 19};

        for (index_type i = 0; i < 8; i++)
        {
          E_values_expected[i] *= multiplier;
        }

        real_type tol = 1e-8;
        for (index_type j = 0; j < E->getNumRows() + 1; j++)
        {
          if (fabs(E->getRowData(memory::HOST)[j] - E_row_ptr_expected[j]) > tol)
          {
            std::cout << "Row pointer mismatch at index " << j << ": got "
                      << E->getRowData(memory::HOST)[j] << ", expected " << E_row_ptr_expected[j] << "\n";
            is_correct = false;
          }
        }

        for (index_type j = 0; j < E->getNnz(); j++)
        {
          if (fabs(E->getColData(memory::HOST)[j] - E_col_ind_expected[j]) > tol)
          {
            std::cout << "Column index mismatch at index " << j << ": got "
                      << E->getColData(memory::HOST)[j] << ", expected " << E_col_ind_expected[j] << "\n";
            is_correct = false;
          }
        }

        for (index_type j = 0; j < E->getNnz(); j++)
        {
          if (fabs(E->getValues(memory::HOST)[j] - E_values_expected[j]) > tol)
          {
            std::cout << "Value mismatch at index " << j << ": got "
                      << E->getValues(memory::HOST)[j] << ", expected " << E_values_expected[j] << "\n";
            is_correct = false;
          }
        }

        delete[] E_row_ptr_expected;
        delete[] E_col_ind_expected;
        delete[] E_values_expected;

        return is_correct;
      }

    }; // class HykktCholeskyTests
  } // namespace tests
} // namespace ReSolve
