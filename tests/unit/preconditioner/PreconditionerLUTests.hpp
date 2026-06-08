/**
 * @file PreconditionerLUTests.hpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief Tests for PreconditionerILU class
 *
 */

#pragma once

#include <cmath>

#include <resolve/LinSolverDirect.hpp>
#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <resolve/PreconditionerLU.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Unit tests for PreconditionerLU.
     *
     */
    class PreconditionerLUTests : public TestBase
    {
    public:
      PreconditionerLUTests(memory::MemorySpace memspace, LinSolverDirect* lu_solver)
      {
        memspace_  = memspace;
        lu_solver_ = lu_solver;
      }

      virtual ~PreconditionerLUTests()
      {
      }

      /**
       * @brief Solve a 3x3 diagonal system through PreconditionerLU
       *
       */
      TestOutcome solve()
      {
        TestStatus  status;
        std::string testname(__func__);

        const index_type n   = 3;
        const index_type nnz = 3;

        matrix::Csr* A           = new matrix::Csr(n, n, nnz);
        index_type   row_data[4] = {0, 1, 2, 3};
        index_type   col_data[3] = {0, 1, 2};
        real_type    val_data[3] = {4.0, 5.0, 6.0};

        A->allocateAll(memspace_);
        A->copyFromExternal(row_data, col_data, val_data, memory::HOST, memory::HOST);

        if (memspace_ == memory::DEVICE)
        {
          A->syncData(memory::DEVICE);
        }

        PreconditionerLU precond(lu_solver_);
        precond.setup(A);

        real_type       rhs_data[3] = {4.0, 10.0, 18.0};
        vector::Vector* rhs         = new vector::Vector(n);
        rhs->allocate(memspace_);
        rhs->copyFromExternal(rhs_data, memory::HOST, memspace_);

        vector::Vector* x = new vector::Vector(n);
        x->allocateAll(memspace_);

        precond.apply(rhs, x);

        if (memspace_ == memory::DEVICE)
        {
          x->syncData(memory::HOST);
        }

        const real_type expected[3] = {1.0, 2.0, 3.0};
        const real_type tol         = 1e-12;
        for (index_type i = 0; i < n; ++i)
        {
          if (std::fabs(x->getData(memory::HOST)[i] - expected[i]) > tol)
          {
            status *= false;
            std::cout << testname << ": x[" << i << "] = " << x->getData(memory::HOST)[i]
                      << ", expected " << expected[i] << "\n";
          }
        }

        delete A;
        delete rhs;
        delete x;

        return status.report(testname.c_str());
      }

      TestOutcome checkSide()
      {
        TestStatus  status;
        std::string testname(__func__);

        PreconditionerLU precond(lu_solver_);

        if (precond.getSide() != Preconditioner::Side::RIGHT)
        {
          status *= false;
          std::cout << "Default side should be Side::RIGHT\n";
        }

        precond.setSide(Preconditioner::Side::LEFT);
        if (precond.getSide() != Preconditioner::Side::LEFT)
        {
          status *= false;
          std::cout << "After setSide(Side::LEFT), side was not updated\n";
        }

        return status.report(testname.c_str());
      }

    private:
      memory::MemorySpace memspace_;
      LinSolverDirect*    lu_solver_;
    };

  } // namespace tests
} // namespace ReSolve
