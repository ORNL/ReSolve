
#pragma once

#include <algorithm>
#include <cholmod.h>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
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
     * @brief Tests for class hykkt::CholeskySolver
     *
     */
    class HykktCholeskyTests : public TestBase
    {
    public:
      HykktCholeskyTests(memory::MemorySpace memspace, MatrixHandler& matrixHandler)
        : memspace_(memspace), matrixHandler_(matrixHandler)
      {
        cholmod_start(&Common);
      }

      virtual ~HykktCholeskyTests()
      {
        cholmod_finish(&Common);
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

        index_type   n             = 3;
        matrix::Csr* A             = new matrix::Csr(n, n, 9);
        index_type   A_row_data[4] = {0, 3, 6, 9};
        index_type   A_col_data[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        real_type    A_values[9]   = {4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0};
        A->copyDataFrom(A_row_data, A_col_data, A_values, memory::HOST, memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          A->syncData(memory::DEVICE);
        }

        ReSolve::hykkt::CholeskySolver solver(memspace_);
        solver.addMatrixInfo(A);
        solver.symbolicAnalysis();
        solver.setPivotTolerance(1e-12);
        solver.numericalFactorization();
        vector::Vector* x = new vector::Vector(3);
        x->allocate(memspace_);
        vector::Vector* b         = new vector::Vector(3);
        real_type       b_data[3] = {-6.0, -17.25, 30.0};
        b->copyDataFrom(b_data, memory::HOST, memspace_);
        solver.solve(x, b);

        if (memspace_ == memory::DEVICE)
        {
          x->syncData(memory::HOST);
        }

        real_type expected_x[3] = {1.0, -0.5, 0.25};

        real_type tol = 1e-12;
        for (index_type i = 0; i < n; ++i)
        {
          if (fabs(x->getData(memory::HOST)[i] - expected_x[i]) > tol)
          {
            std::cout << "Test failed at index " << i << ": expected "
                      << expected_x[i] << ", got "
                      << x->getData(memory::HOST)[i] << "\n";
            status *= false;
          }
        }

        delete A;
        delete x;
        delete b;

        return status.report(testname.c_str());
      }

      TestOutcome randomized(index_type n)
      {
        TestStatus  status;
        std::string testname(__func__);
        testname += " n = " + std::to_string(n);

        cholmod_sparse* L      = randomSparseLowerTriangular((size_t) n);
        cholmod_sparse* L_tr   = cholmod_transpose(L, 1, &Common);
        cholmod_sparse* L_times_L_tr = cholmod_ssmult(L, L_tr, 0, 1, 0, &Common);

        matrix::Csr* A = new matrix::Csr((index_type) L_times_L_tr->nrow, 
        (index_type) L_times_L_tr->ncol, 
        (index_type) L_times_L_tr->nzmax);
        A->copyDataFrom(
            static_cast<int*>(L_times_L_tr->p), 
            static_cast<int*>(L_times_L_tr->i), 
            static_cast<double*>(L_times_L_tr->x), 
            memory::HOST, 
            memspace_);

        ReSolve::hykkt::CholeskySolver solver(memspace_);

        // Add A to the solver, symbolic analysis, and numerical factorization
        solver.addMatrixInfo(A);
        solver.symbolicAnalysis();
        solver.numericalFactorization();

        // Generate a random vector x_expected and compute b = A * x_expected
        vector::Vector* x_expected = randomVector(n);

        vector::Vector* b = new vector::Vector(n);
        b->allocate(memspace_);
        real_type alpha = 1.0;
        real_type beta  = 0.0;
        matrixHandler_.matvec(A, x_expected, b, &alpha, &beta, memspace_);

        // Solve the system A * x = b
        vector::Vector* x = new vector::Vector(n);
        x->allocate(memspace_);
        solver.solve(x, b);

        if (memspace_ == memory::DEVICE)
        {
          x->syncData(memory::HOST);
        }

        // Verify result
        real_type tol = 1e-12;
        for (index_type j = 0; j < n; ++j)
        {
          if (fabs(x->getData(memory::HOST)[j] - x_expected->getData(memory::HOST)[j]) > tol)
          {
            printf("Test failed at index %d: expected %.12f, got %.12f\n, difference %.12f\n",
                   j,
                   x_expected->getData(memory::HOST)[j],
                   x->getData(memory::HOST)[j],
                   fabs(x->getData(memory::HOST)[j] - x_expected->getData(memory::HOST)[j]));
            status *= false;
          }
        }

        cholmod_free_sparse(&L, &Common);
        cholmod_free_sparse(&L_tr, &Common);
        cholmod_free_sparse(&L_times_L_tr, &Common);

        delete A;
        delete x_expected;
        delete b;
        delete x;

        return status.report(testname.c_str());
      }

      TestOutcome randomizedReuseSparsityPattern(index_type n, index_type trials)
      {
        TestStatus  status;
        std::string testname(__func__);
        testname += " n = " + std::to_string(n) + ", trials = " + std::to_string(trials);

        cholmod_sparse* L      = randomSparseLowerTriangular((size_t) n);
        cholmod_sparse* L_tr   = cholmod_transpose(L, 1, &Common);
        cholmod_sparse* A_chol = cholmod_ssmult(L, L_tr, 0, 1, 0, &Common);

        matrix::Csr* A = new matrix::Csr((index_type) A_chol->nrow, (index_type) A_chol->ncol, (index_type) A_chol->nzmax);
        A->copyDataFrom(
            static_cast<int*>(A_chol->p), static_cast<int*>(A_chol->i), static_cast<double*>(A_chol->x), memory::HOST, memspace_);

        ReSolve::hykkt::CholeskySolver solver(memspace_);
        for (index_type i = 0; i < trials; ++i)
        {
          // Only do symbolic analysis the first iteration
          solver.addMatrixInfo(A);
          if (i == 0)
          {
            solver.symbolicAnalysis();
          }
          solver.numericalFactorization();

          // Generate a random vector x_expected and compute b = A * x_expected
          vector::Vector* x_expected = randomVector(n);

          vector::Vector* b = new vector::Vector(n);
          b->allocate(memspace_);
          b->setToZero(memspace_);
          real_type alpha = 1.0;
          real_type beta  = 0.0;
          matrixHandler_.matvec(A, x_expected, b, &alpha, &beta, memspace_);

          // Solve the system A * x = b
          vector::Vector* x = new vector::Vector(n);
          x->allocate(memspace_);
          solver.solve(x, b);

          if (memspace_ == memory::DEVICE)
          {
            // x_expected->syncData(memory::HOST);
            x->syncData(memory::HOST);
          }

          // Verify result
          real_type tol = 1e-12;
          for (index_type j = 0; j < n; ++j)
          {
            if (fabs(x->getData(memory::HOST)[j] - x_expected->getData(memory::HOST)[j]) > tol)
            {
              printf("Test failed at index %d: expected %.12f, got %.12f\n, difference %.12f\n",
                     j,
                     x_expected->getData(memory::HOST)[j],
                     x->getData(memory::HOST)[j],
                     fabs(x->getData(memory::HOST)[j] - x_expected->getData(memory::HOST)[j]));
              status *= false;
            }
          }

          // reset values
          for (size_t j = 0; j < L->nzmax; j++)
          {
            static_cast<double*>(L->x)[j] = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
          }
          cholmod_free_sparse(&L_tr, &Common);
          cholmod_free_sparse(&A_chol, &Common);
          L_tr   = cholmod_transpose(L, 1, &Common);
          A_chol = cholmod_ssmult(L, L_tr, 0, 1, 0, &Common);
          A->copyValues(static_cast<double*>(A_chol->x), memory::HOST, memspace_);
          A->setUpdated(memspace_);
          matrixHandler_.setValuesChanged(true, memspace_);

          delete b;
          delete x;
          delete x_expected;
        }

        delete A;
        cholmod_free_sparse(&L, &Common);

        return status.report(testname.c_str());
      }

    private:
      ReSolve::memory::MemorySpace memspace_;
      MatrixHandler&               matrixHandler_;

      cholmod_common Common;

      cholmod_sparse* randomSparseLowerTriangular(size_t n)
      {
        double              density = 2.0 / (double) n;
        size_t              nnz     = 0;
        std::vector<int>    L_p(n + 1, 0);
        std::vector<int>    L_i;
        std::vector<double> L_x;
        for (size_t i = 0; i < n; ++i)
        {
          L_p[i + 1] = L_p[i];
          for (size_t j = i; j < n; ++j)
          {
            // with probability 'density', add a non-zero entry
            if (i == j || static_cast<double>(rand()) / RAND_MAX < density)
            {
              L_i.push_back((int) j);
              // force diagonal entry to be non-zero
              double value = 2.0;
              if (i != j)
              {
                value = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
              }
              L_x.push_back(value);
              L_p[i + 1]++;
              nnz++;
            }
          }
        }

        cholmod_sparse* L = cholmod_allocate_sparse(
            n, n, nnz, 1, 1, 0, CHOLMOD_REAL, &Common);
        std::copy(L_p.begin(), L_p.end(), static_cast<int*>(L->p));
        std::copy(L_i.begin(), L_i.end(), static_cast<int*>(L->i));
        std::copy(L_x.begin(), L_x.end(), static_cast<double*>(L->x));

        return L;
      }

      // matrix::Csr* randomSparseSPDMatrix(size_t n, double density)
      // {
      //   cholmod_sparse* L = randomSparseLowerTriangular(n);
      //   cholmod_sparse* L_tr = cholmod_transpose(L, 1, &Common);

      //   cholmod_sparse* A_chol = cholmod_ssmult(L, L_tr, 0, 1, 0, &Common);

      //   matrix::Csr* A = new matrix::Csr((index_type) A_chol->nrow, (index_type) A_chol->ncol, (index_type) A_chol->nzmax);
      //   A->copyDataFrom(
      //       static_cast<int*>(A_chol->p), static_cast<int*>(A_chol->i), static_cast<double*>(A_chol->x), memory::HOST, memspace_);

      //   cholmod_free_sparse(&L, &Common);
      //   cholmod_free_sparse(&L_tr, &Common);
      //   cholmod_free_sparse(&A_chol, &Common);

      //   return A;
      // }

      vector::Vector* randomVector(index_type n)
      {
        vector::Vector* v = new vector::Vector(n);
        v->allocate(memory::HOST);
        for (index_type i = 0; i < n; ++i)
        {
          v->getData(memory::HOST)[i] = static_cast<double>(rand()) / RAND_MAX;
        }
        v->setDataUpdated(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          v->syncData(memory::DEVICE);
        }
        return v;
      }
    }; // class HykktCholeskyTests
  } // namespace tests
} // namespace ReSolve
