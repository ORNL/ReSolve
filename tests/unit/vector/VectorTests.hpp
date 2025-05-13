#pragma once
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>
#include <resolve/Common.hpp>

namespace ReSolve { 
  namespace tests {
    /**
     * @class Tests for vector handler
     *
     */
    class VectorTests : TestBase
    {
      public:       
        VectorTests(memory::MemorySpace memspace = memory::HOST)
          : memspace_(memspace)
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

        TestOutcome vectorSetToConst(index_type N = 4)
        {
          using constants::ZERO;
          using constants::ONE;

          TestStatus success;
          success = true;

          index_type vector_size = N;
          index_type number_vectors = 3;

          vector::Vector x(vector_size, number_vectors);

          x.setToZero(memspace_);
          success *= verifyAnswer(x, ZERO);

          x.setToConst(1, ONE, memspace_); // set vector 1 to ones
          success *= verifyAnswer(vector_size, x.getVectorData(0, memspace_), ZERO);
          success *= verifyAnswer(vector_size, x.getVectorData(1, memspace_), ONE);
          success *= verifyAnswer(vector_size, x.getVectorData(2, memspace_), ZERO);

          x.setToConst(ONE, memspace_);
          success *= verifyAnswer(x, ONE);

          x.setToZero(1, memspace_); // set vector 1 to zeros
          success *= verifyAnswer(vector_size, x.getVectorData(0, memspace_), ONE);
          success *= verifyAnswer(vector_size, x.getVectorData(1, memspace_), ZERO);
          success *= verifyAnswer(vector_size, x.getVectorData(2, memspace_), ONE);

          return success.report(__func__);
        }

        TestOutcome vectorSyncData(index_type N = 4)
        {
          using constants::ZERO;
          using constants::ONE;

          TestStatus success;
          success = true;

          if (memspace_ == memory::HOST) {
            return success.report(__func__);
          }

          index_type vector_size = N;
          index_type number_vectors = 3;

          vector::Vector x(vector_size, number_vectors);

          // Set all vectors in x on device to ones
          x.setToConst(ONE, memspace_);
          // Sync host (all ones on the host, as well)
          x.syncData(memory::HOST);
          // Set vector 1 to all zeros on host
          x.setToZero(1, memory::HOST);
          // Sync vector 1 on device
          x.syncData(1, memspace_);

          // Check what we have on device now is correct
          success *= verifyAnswer(vector_size, x.getVectorData(0, memspace_), ONE);
          success *= verifyAnswer(vector_size, x.getVectorData(1, memspace_), ZERO);
          success *= verifyAnswer(vector_size, x.getVectorData(2, memspace_), ONE);

          return success.report(__func__);
        }    


      private:
        ReSolve::memory::MemorySpace memspace_{memory::HOST};
        MemoryHandler mh_;

        /// Check if vector elements are set to the same number
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

        /// Check if an array elements are set to the same number
        bool verifyAnswer(index_type size, const real_type* data, real_type answer)
        {
          bool success = true;
          real_type* x = nullptr;

          // If the data is on device copy it to the host
          if (memspace_ == memory::DEVICE) {
            mh_.allocateArrayOnHost(&x, size);
            mh_.copyArrayDeviceToHost(x, data, size);
            // Set `data` to point to the host copy
            data = x;
          }

          for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
            if (!isEqual(data[i], answer)) {
              std::cout << std::setprecision(16);
              success = false;
              std::cout << "Solution vector element x[" << i << "] = " << data[i]
                << ", expected: " << answer << "\n";
              break; 
            }
          }

          if (memspace_ == memory::DEVICE) {
            mh_.deleteOnHost(x);
            data = nullptr;
          }

          return success;
        }
    };//class
  } // namespace tests
} //namespace ReSolve

