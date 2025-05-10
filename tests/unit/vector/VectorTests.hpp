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
          TestStatus success;
          success = true;

          index_type vector_size = 4;
          index_type number_vectors = 3;

          vector::Vector x(vector_size, number_vectors);

          success *= (vector_size == x.getSize());
          success *= (number_vectors == x.getNumVectors());

          return success.report(__func__);
        }

        TestOutcome vectorSetToConstTest(index_type N)
        {
          TestStatus success;
          success = true;

          return success.report(__func__);
        }

        TestOutcome vectorSyncDataTest(index_type N)
        {
          TestStatus success;

          return success.report(__func__);
        }    

        TestOutcome vectorGetDataTest(index_type N)
        {
          TestStatus success;
          success = true;


          return success.report(__func__);
        }    

      private:
        ReSolve::memory::MemorySpace memspace_{memory::HOST};

        // we can verify through norm but that would defeat the purpose of testing vector handler ...
        bool verifyAnswer(vector::Vector& x, real_type answer)
        {
          bool success = true;

          if (memspace_ == memory::DEVICE) {
            x.syncData(memory::HOST);
          }

          for (index_type i = 0; i < x.getSize(); ++i) {
            // std::cout << x->getData("cpu")[i] << "\n";
            if (!isEqual(x.getData(memory::HOST)[i], answer)) {
              std::cout << std::setprecision(16);
              success = false;
              std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                << ", expected: " << answer << "\n";
              break; 
            }
          }
          return success;
        }
    };//class
  } // namespace tests
} //namespace ReSolve

