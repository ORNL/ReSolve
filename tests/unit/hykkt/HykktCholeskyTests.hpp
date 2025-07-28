
#pragma once

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <tests/unit/TestBase.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/hykkt/cholesky/CholeskySolver.hpp>


namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Tests for class hykkt::CholeskySolver
     *
     */
    class HykktCholeskyTests : public TestBase
    {
    public:
      HykktCholeskyTests(memory::MemorySpace memspace = memory::HOST)
      {
        memspace_ = memspace;
      }

      virtual ~HykktCholeskyTests()
      {
      }

      /**
       * @brief Test the solver on a dense 3x3 matrix with known solution.
       *
       * @return TestOutcome the outcome of the test
       */
      TestOutcome minimalCorrectness()
      {
        TestStatus  status;
        std::string testname(__func__);

        index_type n = 3;
        matrix::Csr* A = new matrix::Csr(n, n, 9);
        index_type A_row_data[4] = {0, 3, 6, 9};
        index_type A_col_data[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        real_type A_values[9] = {4.0, 12.0, -16.0,
                                  12.0, 37.0, -43.0,
                                  -16.0, -43.0, 98.0};
        A->copyDataFrom(A_row_data, A_col_data, A_values, memory::HOST, memspace_);

        ReSolve::hykkt::CholeskySolver solver(memspace_);
        solver.addMatrixInfo(A);
        solver.symbolicAnalysis();
        solver.setPivotTolerance(1e-12);
        solver.numericalFactorization();
        vector::Vector* x = new vector::Vector(3);
        x->allocate(memspace_);
        vector::Vector* b = new vector::Vector(3);
        real_type b_data[3] = {-6.0, -17.25, 30.0};
        b->copyDataFrom(b_data, memory::HOST, memspace_);
        solver.solve(x, b);

        if (memspace_ == memory::DEVICE)
        {
          x->syncData(memory::HOST);
        }

        real_type expected_x[3] = {1.0, -0.5, 0.25};
        
        real_type tol = 1e-8;
        for (index_type i = 0; i < n; ++i)
        {
          if (fabs(x->getData(memory::HOST)[i] - expected_x[i]) > tol)
          {
            std::cout << "Test failed at index " << i << ": expected " 
                  << expected_x[i] << ", got " 
                  << x->getData(memory::HOST)[i] << "\n";
            status = FAIL;
          }
        }

        delete A;
        delete x;
        delete b;

        return status.report(testname.c_str());
      }

    private:
      MemoryHandler                mem_;
      ReSolve::memory::MemorySpace memspace_;
    }; // class HykktCholeskyTests
  } // namespace tests
} // namespace ReSolve
