
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

        index_type* A_row_ptr = new index_type[4]{0, 1, 3, 5};
        index_type* A_col_ind = new index_type[5]{0, 1, 2, 0, 2};
        real_type*  A_values  = new real_type[5]{1.0, 2.0, -4.0, 5.0, 3.0};

        index_type* B_row_ptr = new index_type[4]{0, 1, 3, 5};
        index_type* B_col_ind = new index_type[5]{0, 0, 1, 1, 2};
        real_type*  B_values  = new real_type[5]{-1.0, -2.0, 3.0, 2.0, 4.0};

        index_type* D_row_ptr = new index_type[4]{0, 2, 4, 6};
        index_type* D_col_ind = new index_type[6]{0, 2, 0, 1, 1, 2};
        real_type*  D_values  = new real_type[6]{3.0, -4.0, -2.0, 1.0, 5.0, 7.0};

        matrix::Csr* A = new matrix::Csr(3, 3, 5);
        matrix::Csr* B = new matrix::Csr(3, 3, 5);
        matrix::Csr* D = new matrix::Csr(3, 3, 6);

        A->copyDataFrom(A_row_ptr, A_col_ind, A_values, memory::HOST, memspace_);
        B->copyDataFrom(B_row_ptr, B_col_ind, B_values, memory::HOST, memspace_);
        D->copyDataFrom(D_row_ptr, D_col_ind, D_values, memory::HOST, memspace_);

        matrix::Csr* E = nullptr;

        spgemm.addProductMatrices(A, B);
        spgemm.addSumMatrix(D);
        spgemm.addResultMatrix(&E);
        spgemm.compute();

        if (memspace_ == memory::DEVICE)
        {
          E->syncData(memory::HOST);
        }

        index_type* E_row_ptr_expected = new index_type[4]{0, 2, 5, 8};
        index_type* E_col_ind_expected = new index_type[8]{0, 2, 0, 1, 2, 0, 1, 2};
        real_type*  E_values_expected  = new real_type[8]{2.0, -4.0, -6.0, -1.0, -16, -5.0, 11, 19};

        real_type tol = 1e-8;
        for (index_type j = 0; j < E->getNumRows() + 1; j++)
        {
          if (fabs(E->getRowData(memory::HOST)[j] - E_row_ptr_expected[j]) > tol)
          {
            std::cout << "Row pointer mismatch at index " << j << ": got "
                      << E->getRowData(memory::HOST)[j] << ", expected " << E_row_ptr_expected[j] << "\n";
            status *= false;
          }
        }

        for (index_type j = 0; j < E->getNnz(); j++)
        {
          if (fabs(E->getColData(memory::HOST)[j] - E_col_ind_expected[j]) > tol)
          {
            std::cout << "Column index mismatch at index " << j << ": got "
                      << E->getColData(memory::HOST)[j] << ", expected " << E_col_ind_expected[j] << "\n";
            status *= false;
          }
        }

        for (index_type j = 0; j < E->getNnz(); j++)
        {
          if (fabs(E->getValues(memory::HOST)[j] - E_values_expected[j]) > tol)
          {
            std::cout << "Value mismatch at index " << j << ": got "
                      << E->getValues(memory::HOST)[j] << ", expected " << E_values_expected[j] << "\n";
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

    private:
      ReSolve::memory::MemorySpace memspace_;

    }; // class HykktCholeskyTests
  } // namespace tests
} // namespace ReSolve
