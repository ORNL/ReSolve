
/**
 * @file UnitTest.hpp
 *
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */
#pragma once

#define THROW_NULL_DEREF throw std::runtime_error("error")

#include <limits>
#include <cmath>
#include <iostream>

namespace ReSolve { namespace tests {

struct Result
{
  Result(){}
  ~Result(){}
  Result(const Result& r)
  {
    this->success += r.success;    
    this->failure += r.failure;
    this->skip    += r.skip;
    this->expected_failure   += r.expected_failure;
    this->unexpected_success += r.unexpected_success;
  }

  void init()
  {
    this->success = 0;    
    this->failure = 0;
    this->skip    = 0;
    this->expected_failure   = 0;
    this->unexpected_success = 0;
  }

  void pass(bool isPass)
  {
    this->init();
    if(isPass)
      this->success = 1;
    else
      this->failure = 1;
  }

  int success = 0;
  int failure = 0;
  int skip = 0;
  int expected_failure = 0;
  int unexpected_success = 0;

  Result& operator+=(const Result& rhs)
  {
    this->success += rhs.success;    
    this->failure += rhs.failure;
    this->skip    += rhs.skip;
    this->expected_failure   += rhs.expected_failure;
    this->unexpected_success += rhs.unexpected_success;
    
    return *this;
  }

};

Result operator+(const Result& lhs, const Result& rhs)
{
  return Result(lhs) += rhs;
}

Result& skipTest(Result& r)
{
  r.success = 0;
  r.failure = 0;
  r.skip = 1;
  r.expected_failure = 0;
  r.unexpected_success = 0;
  return r;
}

Result& expectTestFailure(Result& r)
{
  if(r.failure == 1)
  {
    r.expected_failure = 1;
    r.failure = 0;
  }
  else if(r.success == 1)
  {
    r.unexpected_success = 1;
    r.success = 0;
  }
  else
  {
    std::cout << "WARNING: Miscounted test sucesses/failures ...\n";
    r.success = 0;
    r.failure = 0;
    r.skip = 0;
    r.expected_failure = 0;
    r.unexpected_success = 0;
  }
  return r;
}


using real_type             = double;
using local_ordinal_type    = int;
using global_ordinal_type   = int;

static const real_type zero = 0.0;
static const real_type quarter = 0.25;
static const real_type half = 0.5;
static const real_type one = 1.0;
static const real_type two = 2.0;
static const real_type three = 3.0;
static const real_type eps = 10*std::numeric_limits<real_type>::epsilon();
static const int SKIP_TEST = -1;

// must be const pointer and const dest for
// const string declarations to pass
// -Wwrite-strings
static const char * const  RED       = "\033[1;31m";
static const char * const  GREEN     = "\033[1;32m";
static const char * const  YELLOW    = "\033[1;33m";
static const char * const  ORANGE    = "\033[31;1m";
static const char * const  CLEAR     = "\033[0m";

class TestBase
{
public:
  TestBase()
    : mem_space_("DEFAULT")
  {
  }
  inline void set_mem_space(const std::string& mem_space)
  {
    mem_space_ = mem_space;
  }
  inline std::string get_mem_space() const
  {
    return mem_space_;
  }
protected:
  /// Returns true if two real numbers are equal within tolerance
  [[nodiscard]] static
  bool isEqual(const real_type a, const real_type b)
  {
    return (std::abs(a - b)/(1.0 + std::abs(b)) < eps);
  }

  /// Prints error output for each rank
  static void printMessage(const bool pass, const char* funcname, const int rank=0)
  {
    if(pass)
    {
      if(rank == 0)
      {
        std::cout << "--- " << GREEN << "PASS" << CLEAR << ": Test " << funcname << "\n";
      }
    }
    // else if (fail == SKIP_TEST)
    // {
    //   if(rank == 0)
    //   {
    //     std::cout << YELLOW << "--- SKIP: Test " << funcname << CLEAR << "\n";
    //   }
    // }
    else
    {
      if(rank == 0)
      {
        std::cout << "--- " << RED << "FAIL" << CLEAR << ": Test " << funcname << "\n";
        // std::cout << RED << "--- FAIL: Test " << funcname << " on rank " << rank << CLEAR << "\n";
      }
    }
  }
protected:
  std::string mem_space_;
};

}} // namespace ReSolve::tests