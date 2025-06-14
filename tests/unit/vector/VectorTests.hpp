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
#include <resolve/MemoryUtils.hpp>

namespace ReSolve {
  namespace tests {
    /**
     * @class Tests for vector operations.
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

        /**
         * @brief Test vector constructor with specified size and number of vectors.
         *
         * @param[in] N Number of elements in the vector.
         * @param[in] k Number of vectors in the multivector.
         * @return TestOutcome indicating success or failure of the test.
         */
        TestOutcome vectorConstructor(index_type N, index_type k)
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

        /**
         * @brief Test vector constructor with specified size and default number of vectors (1).
         *
         * @param[in] N Number of elements in the vector.
         * @return TestOutcome indicating success or failure of the test.
         */
        TestOutcome vectorConstructor(index_type N) {
          return vectorConstructor(N, 1);
        }

        /**
         * @brief Test resizing a vector to a new size.
         *
         * @param[in] N Current size of the vector.
         * @param[in] new_N New size to which the vector should be resized.
         * @return TestOutcome indicating success or failure of the test.
         */
        TestOutcome resize(index_type N, index_type new_N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          x.allocate(memspace_);

          x.resize(new_N);

          if (x.getSize() != new_N) {
            std::cout << "The size of the vector after resizing is " << x.getSize()
                    << ", expected: " << new_N << "\n";
            status *= false;
          }

          return status.report(__func__);
        }

        /**
         * @brief Test setting data in a vector from array.
         *
         * @param[in] N Number of elements in the vector.
         * @return TestOutcome indicating success or failure of the test.
         */
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

          delete[] data;
          return status.report(__func__);
        }

        /**
         * @brief Test copying data between vector-array and vector-vector.
         *
         * This creates an array, copies it to a vector in the current memory space, then
         * copies it to another vector in the same memory space, and finally back to a third on the
         * HOST. Then, it verifies the content of the final vector. This test only passes if all
         * copies are successful.
         *
         * @param[in] N Number of elements in the vector.
         * @return TestOutcome indicating success or failure of the test.
         */
        TestOutcome copyDataFrom(index_type N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }

          // array -> memspace
          x.copyDataFrom(data, memory::HOST, memspace_);

          // memspace -> memspace
          vector::Vector y(N);
          y.copyDataFrom(&x, memspace_, memspace_);

          // memspace -> host
          vector::Vector z(N);
          z.copyDataFrom(&y, memspace_, memory::HOST);

          real_type* z_data = z.getData(memory::HOST);

          if (z_data == nullptr) {
            std::cout << "The data pointer is null after copying from vector.\n";
            status *= false;
          } else {
            for (int i = 0; i < N; ++i) {
              if (!isEqual(z_data[i], data[i])) {
                std::cout << "The data in the copied vector is incorrect at index " << i
                          << ", expected: " << data[i]
                          << ", got: " << z_data[i] << "\n";
                status *= false;
                break;
              }
            }
          }

          delete[] data;
          return status.report(__func__);
        }

        /**
         * @brief Test copying data from vector to an array.
         *
         * This creates a vector, copies data to it, and then copies the data to an array
         * in the current memory space. Finally, it uses the MemoryHandler to copy the data
         * to HOST for verification.
         *
         * @param[in] N Number of elements in the vector.
         * @return TestOutcome indicating success or failure of the test.
         */
        TestOutcome copyDataTo(index_type N)
        {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }

          x.copyDataFrom(data, memory::HOST, memspace_);

          // Copy data to an array on current memspace
          real_type* dest = new real_type[N];
          // second argument is in/out
          x.copyDataTo(dest, memspace_);

          // Copy to host to verify
          real_type* dest_h = new real_type[N];
          if (memspace_ == memory::DEVICE) {
            mem_.copyArrayDeviceToHost(dest_h, dest, N);
            dest = dest_h;
          } else {
            // If we are on HOST, we can use dest directly
            delete[] dest_h;
            dest_h = dest;
          }

          // Verify the copied data
          for (int i = 0; i < N; ++i) {
            if (!isEqual(dest_h[i], data[i])) {
              std::cout << "The data in the destination array is incorrect at index " << i
                        << ", expected: " << data[i]
                        << ", got: " << dest_h[i] << "\n";
              status *= false;
              break;
            }
          }

          delete[] data;
          delete[] dest;
          return status.report(__func__);
        }

        /**
         * @brief Test setting all elements of a vector to a constant value.
         *
         * @param[in] N Number of elements in the vector.
         * @param[in] constValue The constant value to set all elements to.
         * @return TestOutcome indicating success or failure of the test.
         */
        TestOutcome setToConst(index_type N, real_type constValue) {
          TestStatus status;
          status = true;

          vector::Vector x(N);
          x.allocate(memspace_);

          // Set all elements to a constant value
          if (isEqual(constValue, (real_type) 0.0)) {
            x.setToZero(memspace_);
          } else {
            x.setToConst(constValue, memspace_);
          }

          real_type* x_data = x.getData(memspace_);

          if (x_data == nullptr) {
            std::cout << "The data pointer is null after setting to constant.\n";
            status *= false;
          } else {
            for (int i = 0; i < N; ++i) {
              if (!isEqual(x_data[i], constValue)) {
                std::cout << "The data in the vector is incorrect at index " << i
                          << ", expected: " << constValue
                          << ", got: " << x_data[i] << "\n";
                status *= false;
                break;
              }
            }
          }

          return status.report(__func__);
        }

        /**
         * @brief Test syncing data between HOST and DEVICE memory spaces.
         *
         * Creates a vector allocated in the specified memory space, then sync
         * to the other memory space and verify that the data is synced correctly.
         *
         * @param[in] N Number of elements in the vector.
         * @param[in] memspaceFrom Memory space from which to sync data (HOST or DEVICE).
         * @return TestOutcome indicating success or failure of the test.
         */
        TestOutcome syncData(index_type N, memory::MemorySpace memspaceFrom) {
          memory::MemorySpace memspaceTo = (memspaceFrom == memory::HOST) ? memory::DEVICE : memory::HOST;

          TestStatus status;
          status = true;

          vector::Vector x(N);
          x.allocate(memspaceFrom);

          real_type* data = new real_type[N];
          for (int i = 0; i < N; ++i) {
            data[i] = 0.1 * (real_type) i;
          }
          x.copyDataFrom(data, memory::HOST, memspaceFrom);

          // Sync data between host and device
          x.syncData(memspaceTo);

          // Bring back to host to verify
          vector::Vector x_host(N);
          x_host.copyDataFrom(&x, memspaceTo, memory::HOST);

          real_type* x_synced_data = x_host.getData(memory::HOST);

          if (x_synced_data == nullptr) {
            std::cout << "The data pointer is null after syncing.\n";
            status *= false;
          } else {
            for (int i = 0; i < N; ++i) {
              if (!isEqual(x_synced_data[i], data[i])) {
                std::cout << "The data in the vector after sync is incorrect at index " << i
                          << ", expected: " << data[i]
                          << ", got: " << x_synced_data[i] << "\n";
                status *= false;
              }
            }
          }

          return status.report(__func__);
        }

      private:
        ReSolve::memory::MemorySpace memspace_;
        MemoryHandler mem_;
    };//class
  } // namespace tests
} //namespace ReSolve
