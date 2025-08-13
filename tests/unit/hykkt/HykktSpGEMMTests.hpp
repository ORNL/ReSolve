
#pragma once

#include <algorithm>
#include <cholmod.h>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <tests/unit/TestBase.hpp>

#include <resolve/hykkt/spgemm/SpGEMM.hpp>

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

        return status.report(testname.c_str());
      }

      

    private:
      ReSolve::memory::MemorySpace memspace_;
      
    }; // class HykktCholeskyTests
  } // namespace tests
} // namespace ReSolve