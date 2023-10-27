#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/MemoryUtils.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { namespace tests {

/**
 * @class Unit tests for memory handler class
 */
class MemoryUtilsTests : TestBase
{
public:
  MemoryUtilsTests(std::string memspace) : memspace_(memspace) 
  {}
  virtual ~MemoryUtilsTests()
  {}

  TestOutcome allocateAndDelete()
  {
    TestStatus status;
    status = true;

    MemoryHandler mh;

    index_type n = 1000;
    size_t     m = 8000;
    index_type* i = nullptr;
    real_type*  r = nullptr;

    mh.allocateArrayOnDevice(&i, n);
    mh.allocateBufferOnDevice((void**) &r, m);

    status *= (i != nullptr);
    status *= (r != nullptr);

    mh.deleteOnDevice(i);
    mh.deleteOnDevice(r);

    return status.report(__func__);
  }

  TestOutcome memsetAndMemcpy()
  {
    TestStatus status;
    status = true;

    MemoryHandler mh;

    index_type n = 10;

    real_type zero = 0.0;
    real_type minusone = -1.0;

    // Create raw arrays on the host and set their elements to -1
    real_type* array1 = new real_type[n]{0};
    real_type* array2 = new real_type[n]{0};
    std::fill_n(array1, n, minusone);
    std::fill_n(array2, n, minusone);

    // Allocate arrays of size n on the device
    real_type* devarray1 = nullptr;
    real_type* devarray2 = nullptr;
    mh.allocateArrayOnDevice(&devarray1, n);
    mh.allocateArrayOnDevice(&devarray2, n);

    // Set devarray1 elements to 0 and copy it to array1
    mh.setZeroArrayOnDevice(devarray1, n);
    mh.copyArrayDeviceToHost(array1, devarray1, n);
    status *= verifyAnswer(array1, zero, n);

    // Copy array2 (values -1) to devarray2 and then devarray2 to array1
    mh.copyArrayHostToDevice(devarray2, array2, n);
    mh.copyArrayDeviceToHost(array1, devarray2, n);
    status *= verifyAnswer(array1, minusone, n);

    // Copy devarray1 (values 0) to devarray2 and then to array2
    mh.copyArrayDeviceToDevice(devarray2, devarray1, n);
    mh.copyArrayDeviceToHost(array2, devarray2, n);
    status *= verifyAnswer(array2, zero, n);

    return status.report(__func__);
  }


private:
  std::string memspace_{"cpu"};

  bool verifyAnswer(real_type* x, real_type answer, index_type n)
  {
    bool status = true;

    for (index_type i = 0; i < n; ++i) {
      if (!isEqual(x[i], answer)) {
        status = false;
        std::cout << "Solution vector element x[" << i << "] = " << x[i]
                  << ", expected: " << answer << "\n";
        break; 
      }
    }
    return status;
  }

}; // class MemoryUtilsTests

}} // namespace ReSolve::tests
