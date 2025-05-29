#pragma once
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <tests/unit/TestBase.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

namespace ReSolve { 
  namespace tests {
    /**
     * @class Tests for vector operations.
     *
     */
    class VectorTests : TestBase
    {
      public:
        VectorTests(ReSolve::VectorHandler& handler): handler_(handler) 
        {
          if (handler_.getIsCudaEnabled() || handler_.getIsHipEnabled()) {
            memspace_ = memory::DEVICE;
          } else {
            memspace_ = memory::HOST;
          }
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
        
        TestOutcome dimensions(index_type N, index_type k)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N, k);

          if (x.getCapacity() != N) {
            std::cout << "The capacity of the vector is " << x.getCapacity() 
                    << ", expected: " << N << "\n";
            status *= false;
          }

          if (x.getSize() != N) {
            std::cout << "The size of the vector is " << x.getSize()
                    << ", expected: " << N << "\n";
            status *= false;
          }

          if (x.getNumVectors() != k) {
            std::cout << "The number of vectors in the multivector is " << x.getNumVectors()
                    << ", expected: " << k << "\n";
            status *= false;
          }

          return status.report(__func__);
        }

        TestOutcome resize(index_type N, index_type newN)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          x.allocate(memspace_);
          
          x.resize(newN);

          if (x.getSize() != newN) {
            std::cout << "The size of the vector after resizing is " << x.getSize()
                    << ", expected: " << newN << "\n";
            status *= false;
          }

          return status.report(__func__);
        }
      
        TestOutcome setData(index_type N)
        {
          TestStatus status;
          status = true;
          
          vector::Vector x(N);

          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }
          x.setData(data, memspace_);

          real_type* x_data = x.getData(memspace_);
          
          if (x_data == nullptr) {
            std::cout << "The data pointer is null after setting.\n";
            status *= false;
          } else {
            for (int i = 0; i < N; ++i) {
              if (!isEqual(x_data[i], data[i])) {
                std::cout << "The data in the vector is incorrect at index " << i 
                          << ", expected: " << data[i] 
                          << ", got: " << x_data[i] << "\n";
                status *= false;
                break;
              }
            }
          }

          return status.report(__func__);
        }

        TestOutcome copyDataFromArray(index_type N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }
          
          x.copyDataFrom(data, memory::HOST, memspace_);

          // Modify the original vector to verify that copied data does not change
          for (int i = 0; i < N; ++i) {
            data[i] = 0.2 * (real_type) i;
          }
          
          real_type* x_data = x.getData(memspace_);
          
          if (x_data == nullptr) {
            std::cout << "The data pointer is null after copying.\n";
            status *= false;
          } else {
            for (int i = 0; i < N; ++i) {
              if (!isEqual(x_data[i], 0.1 * (real_type) i)) {
                std::cout << "The data in the vector is incorrect at index " << i 
                          << ", expected: " << 0.1 * (real_type) i
                          << ", got: " << x_data[i] << "\n";
                status *= false;
                break;
              }
            }
          }

          return status.report(__func__);
        }

        TestOutcome copyDataFromVector(index_type N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }
          
          x.copyDataFrom(data, memory::HOST, memspace_);

          // Create another vector and copy data from the first one
          vector::Vector y(N);
          y.copyDataFrom(&x, memory::HOST, memspace_);

          // Modify the original vector to verify that copied data does not change
          for (int i = 0; i < N; ++i) {
            data[i] = 0.2 * (real_type) i;
          }

          real_type* y_data = y.getData(memspace_);
          
          if (y_data == nullptr) {
            std::cout << "The data pointer is null after copying from vector.\n";
            status *= false;
          } else {
            for (int i = 0; i < N; ++i) {
              if (!isEqual(y_data[i], 0.1 * (real_type) i)) {
                std::cout << "The data in the copied vector is incorrect at index " << i 
                          << ", expected: " << 0.1 * (real_type) i
                          << ", got: " << y_data[i] << "\n";
                status *= false;
                break;
              }
            }
          }

          return status.report(__func__);
        }

        TestOutcome copyDataToArray(index_type N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }
          
          x.copyDataFrom(data, memory::HOST, memspace_);

          // Copy data to an array
          real_type* dest = new real_type[N];
          x.copyDataTo(dest, memspace_);

          // Verify the copied data
          for (int i = 0; i < N; ++i) {
            if (!isEqual(dest[i], 0.1 * (real_type) i)) {
              std::cout << "The data in the destination array is incorrect at index " << i 
                        << ", expected: " << 0.1 * (real_type) i
                        << ", got: " << dest[i] << "\n";
              status *= false;
              break;
            }
          }

          delete[] dest;
          return status.report(__func__);
        }

        TestOutcome copyDataToVector(index_type N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }
          
          x.copyDataFrom(data, memory::HOST, memspace_);

          // Copy data to another vector
          vector::Vector y(N);
          y.copyDataFrom(&x, memory::HOST, memspace_);

          // Verify the copied data
          real_type* y_data = y.getData(memspace_);
          
          if (y_data == nullptr) {
            std::cout << "The data pointer is null after copying to vector.\n";
            status *= false;
          } else {
            for (int i = 0; i < N; ++i) {
              if (!isEqual(y_data[i], 0.1 * (real_type) i)) {
                std::cout << "The data in the copied vector is incorrect at index " << i 
                          << ", expected: " << 0.1 * (real_type) i
                          << ", got: " << y_data[i] << "\n";
                status *= false;
                break;
              }
            }
          }

          return status.report(__func__);
        }

      private:
        ReSolve::VectorHandler& handler_;
        ReSolve::memory::MemorySpace memspace_{memory::HOST};
    };//class
  } // namespace tests
} //namespace ReSolve

