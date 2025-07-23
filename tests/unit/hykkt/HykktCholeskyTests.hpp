
#pragma once

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/hykkt/cholesky/CholeskySolver.hpp>
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
      HykktCholeskyTests(memory::MemorySpace memspace = memory::HOST)
      {
        memspace_ = memspace;
      }

      virtual ~HykktCholeskyTests()
      {
      }

      TestOutcome test()
      {
        ReSolve::hykkt::CholeskySolver solver(memspace_);

        return PASS;
      }

    private:
      MemoryHandler                mem_;
      ReSolve::memory::MemorySpace memspace_;
    }; // class HykktCholeskyTests
  } // namespace tests
} // namespace ReSolve
