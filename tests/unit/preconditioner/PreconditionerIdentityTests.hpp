/**
 * @file PreconditionerIdentityTests.hpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief Tests for PreconditionerIdentity class.
 *
 */

#pragma once

#include <iostream>
#include <string>

#include <resolve/PreconditionerIdentity.hpp>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Unit tests for PreconditionerIdentity.
     */
    class PreconditionerIdentityTests : public TestBase
    {
    public:
      explicit PreconditionerIdentityTests(memory::MemorySpace memspace)
        : memspace_(memspace)
      {
      }

      /**
       * @brief Check that setup is independent of a system matrix.
       */
      TestOutcome setup()
      {
        TestStatus status;

        PreconditionerIdentity preconditioner(memspace_);
        status *= (preconditioner.setup(nullptr) == 0);
        status *= (preconditioner.reset(nullptr) == 0);

        return status.report(__func__);
      }

      /**
       * @brief Check the default and configurable preconditioning side.
       */
      TestOutcome checkSide()
      {
        TestStatus status;

        PreconditionerIdentity preconditioner(memspace_);
        status *= (preconditioner.getSide() == Preconditioner::Side::RIGHT);
        status *= (preconditioner.setSide(Preconditioner::Side::LEFT) == 0);
        status *= (preconditioner.getSide() == Preconditioner::Side::LEFT);

        return status.report(__func__);
      }

      /**
       * @brief Check that apply copies the input without changing its values.
       */
      TestOutcome apply()
      {
        TestStatus status;

        const index_type n         = 4;
        const real_type  values[n] = {1.0, -2.0, 3.5, 0.25};
        vector::Vector   rhs(n);
        vector::Vector   x(n);

        rhs.allocate(memspace_);
        x.allocate(memspace_);
        rhs.copyFromExternal(values, memory::HOST, memspace_);

        PreconditionerIdentity preconditioner(memspace_);
        status *= (preconditioner.apply(nullptr, &x) == 1);
        status *= (preconditioner.apply(&rhs, nullptr) == 1);
        status *= (preconditioner.apply(&rhs, &x) == 0);

        if (memspace_ == memory::DEVICE)
        {
          x.syncData(memory::HOST);
        }

        for (index_type i = 0; i < n; ++i)
        {
          if (!isEqual(x.getData(memory::HOST)[i], values[i]))
          {
            std::cout << __func__ << ": x[" << i << "] = "
                      << x.getData(memory::HOST)[i] << ", expected " << values[i] << "\n";
            status *= false;
          }
        }

        return status.report(__func__);
      }

    private:
      memory::MemorySpace memspace_;
    };
  } // namespace tests
} // namespace ReSolve
