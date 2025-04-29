#pragma once
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { 
  namespace tests {
    /**
     * @class Tests for vector handler
     *
     */
    class VectorTests : TestBase
    {
      public:       
        VectorTests()
        {
        }

        virtual ~VectorTests()
        {
        }

        TestOutcome vectorConstructor()
        {
          TestStatus status;
          status.skipTest();

          return status.report(__func__);
        }

        TestOutcome vectorSetToConstTest(index_type N)
        {
          TestStatus status;
          status = true;

          return status.report(__func__);
        }

        TestOutcome vectorSyncDataTest(index_type N)
        {
          TestStatus status;

          return status.report(__func__);
        }    

        TestOutcome vectorGetDataTest(index_type N)
        {
          TestStatus status;
          status = true;


          return status.report(__func__);
        }    

      private:
        ReSolve::memory::MemorySpace memspace_{memory::HOST};

        // we can verify through norm but that would defeat the purpose of testing vector handler ...
        bool verifyAnswer(vector::Vector& x, real_type answer)
        {
          bool status = true;

          if (memspace_ == memory::DEVICE) {
            x.syncData(memory::HOST);
          }

          for (index_type i = 0; i < x.getSize(); ++i) {
            // std::cout << x->getData("cpu")[i] << "\n";
            if (!isEqual(x.getData(memory::HOST)[i], answer)) {
              std::cout << std::setprecision(16);
              status = false;
              std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                << ", expected: " << answer << "\n";
              break; 
            }
          }
          return status;
        }
    };//class
  } // namespace tests
} //namespace ReSolve

