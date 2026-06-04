/**
 * @file PreconditionerUserMatrixTests.hpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief Tests for PreconditionerUserMatrix class.
 *
 */

#pragma once

#include <resolve/PreconditionerUserMatrix.hpp>
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
     * @class Unit tests for PreconditionerUserMatrix.
     *
     */
    class PreconditionerUserMatrixTests : public TestBase
    {
    public:
      PreconditionerUserMatrixTests(memory::MemorySpace memspace, MatrixHandler& handler)
        : memspace_(memspace),
          handler_(handler)
      {
      }

      virtual ~PreconditionerUserMatrixTests()
      {
      }

      /**
       * @brief Test default side is right and setSide works for both enum values.
       *
       */
      TestOutcome checkSide()
      {
        TestStatus  status;
        std::string testname(__func__);

        PreconditionerUserMatrix precond(&handler_);

        if (precond.getSide() != Preconditioner::Side::RIGHT)
        {
          status *= false;
          std::cout << testname << ": default side is not Side::RIGHT\n";
        }

        if (precond.setSide(Preconditioner::Side::LEFT) != 0
            || precond.getSide() != Preconditioner::Side::LEFT)
        {
          status *= false;
          std::cout << testname << ": setSide(Side::LEFT) failed\n";
        }

        if (precond.setSide(Preconditioner::Side::RIGHT) != 0
            || precond.getSide() != Preconditioner::Side::RIGHT)
        {
          status *= false;
          std::cout << testname << ": setSide(Side::RIGHT) failed\n";
        }

        return status.report(testname.c_str());
      }

      /**
       * @brief Test setPrecMatrix(B) stores B and getPrecMatrix() retrieves it.
       */
      TestOutcome setPrecMatrix()
      {
        TestStatus status;
        status = true;

        PreconditionerUserMatrix precond(&handler_);
        matrix::Csr              B(3, 3, 3);

        int ret = precond.setPrecMatrix(&B);
        if (ret != 0)
        {
          std::cout << "setMatrix(B) returned " << ret << ", expected 0\n";
          status *= false;
        }
        if (precond.getPrecMatrix() != &B)
        {
          std::cout << "getPrecMatrix() did not return the matrix passed to setPrecMatrix()\n";
          status *= false;
        }

        return status.report(__func__);
      }

      /**
       * @brief Test apply() performs correct matvec with B.
       *
       * B = diag(2, 3, 4), x = [1, 1, 1] => y = [2, 3, 4].
       */
      TestOutcome apply()
      {
        TestStatus status;
        status = true;

        const index_type n   = 3;
        const index_type nnz = 3;
        matrix::Csr*     B   = new matrix::Csr(n, n, nnz);

        index_type row_data[4] = {0, 1, 2, 3};
        index_type col_data[3] = {0, 1, 2};
        real_type  val_data[3] = {2.0, 3.0, 4.0};

        B->copyFromExternal(row_data, col_data, val_data, memory::HOST, memory::HOST);

        if (memspace_ == memory::DEVICE)
        {
          B->allocateMatrixData(memspace_);
          B->syncData(memspace_);
        }

        vector::Vector x(n);
        x.allocate(memory::HOST);
        x.allocate(memspace_);

        for (index_type i = 0; i < n; ++i)
        {
          x.getData(memory::HOST)[i] = 1.0;
        }

        x.setDataUpdated(memory::HOST);

        if (memspace_ == memory::DEVICE)
        {
          x.syncData(memspace_);
        }

        vector::Vector y(n);
        y.allocate(memory::HOST);
        y.allocate(memspace_);
        y.setToZero(memspace_);

        PreconditionerUserMatrix precond(&handler_);
        precond.setPrecMatrix(B);

        int ret = precond.apply(&x, &y);

        if (ret != 0)
        {
          std::cout << "apply() returned " << ret << ", expected 0\n";
          status *= false;
        }

        if (memspace_ == memory::DEVICE)
        {
          y.syncData(memory::HOST);
        }

        const real_type expected[3] = {2.0, 3.0, 4.0};
        for (index_type i = 0; i < n; ++i)
        {
          if (!isEqual(y.getData(memory::HOST)[i], expected[i]))
          {
            std::cout << "y[" << i << "] = " << y.getData(memory::HOST)[i]
                      << ", expected " << expected[i] << "\n";
            status *= false;
          }
        }

        delete B;
        return status.report(__func__);
      }

    private:
      memory::MemorySpace     memspace_{memory::HOST};
      ReSolve::MatrixHandler& handler_;
    };

  } // namespace tests
} // namespace ReSolve
