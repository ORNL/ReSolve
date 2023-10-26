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


private:
  std::string memspace_{"cpu"};

  // bool verifyAnswer(vector::Vector& x, real_type answer, std::string memspace)
  // {
  //   bool status = true;
  //   if (memspace != "cpu") {
  //     x.copyData(memspace, "cpu");
  //   }

  //   for (index_type i = 0; i < x.getSize(); ++i) {
  //     // std::cout << x.getData("cpu")[i] << "\n";
  //     if (!isEqual(x.getData("cpu")[i], answer)) {
  //       status = false;
  //       std::cout << "Solution vector element x[" << i << "] = " << x.getData("cpu")[i]
  //                 << ", expected: " << answer << "\n";
  //       break; 
  //     }
  //   }
  //   return status;
  // }

}; // class MemoryUtilsTests

}} // namespace ReSolve::tests
