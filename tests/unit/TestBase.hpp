
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

#include <resolve/Common.hpp>

namespace ReSolve { namespace tests {

enum TestOutcome {PASS=0, FAIL, SKIP, EXPECTED_FAIL, UNEXPECTED_PASS};

class TestStatus
{
public:
  TestStatus()
  : outcome_(TestOutcome::PASS)
  {}
  TestStatus(const char* funcname) 
  : outcome_(TestOutcome::PASS),
    funcname_(funcname)
  {}
  ~TestStatus()
  {}

  TestStatus& operator=(const bool isPass)
  {
    if(isPass)
      outcome_ = TestOutcome::PASS;
    else
      outcome_ = TestOutcome::FAIL;   
    return *this;
  }

  TestStatus& operator*=(const bool isPass)
  {
    if(!isPass)
      outcome_ = TestOutcome::FAIL;   
    return *this;
  }

  void skipTest()
  {
    outcome_ = TestOutcome::SKIP;   
  }

  void expectFailure()
  {
    expectFailure_ = true;
  }

  TestOutcome report()
  {
    return report(funcname_);
  }

  TestOutcome report(const char* funcname)
  {
    if (expectFailure_)
    {
      if ((outcome_ == FAIL) || (outcome_ == EXPECTED_FAIL))
        outcome_ = EXPECTED_FAIL;
      else if ((outcome_ == PASS) || (outcome_ == UNEXPECTED_PASS))
        outcome_ = UNEXPECTED_PASS;
      else
        outcome_ = SKIP;
    }

    switch(outcome_)
    {
      using namespace colors;
      case PASS:
        std::cout << "--- " << GREEN << "PASS" << CLEAR << ": Test " << funcname << "\n";
        break;
      case FAIL:
        std::cout << "--- " << RED << "FAIL" << CLEAR << ": Test " << funcname << "\n";
        break;
      case SKIP:
        std::cout << "--- " << YELLOW << "SKIP" << CLEAR << ": Test " << funcname << CLEAR << "\n";
        break;
      case EXPECTED_FAIL:
        std::cout << "--- " << ORANGE << "FAIL" << CLEAR << " (EXPECTED)" << ": Test " << funcname << "\n";
        break;
      case UNEXPECTED_PASS:
        std::cout << "--- " << BLUE << "PASS" << CLEAR << " (UNEXPECTED)" << ": Test " << funcname << "\n";
        break;
      default:
        std::cout << "--- " << RED << "FAIL" << CLEAR << "Unrecognized test result " << outcome_ 
                  << " for test " << funcname << "\n";
    }
    return outcome_;
  }

private:
  TestOutcome outcome_;
  const char* funcname_;
  bool expectFailure_ = false;
};



struct TestingResults
{
  int success = 0;
  int failure = 0;
  int skip = 0;
  int expected_failure = 0;
  int unexpected_success = 0;

  TestingResults(){}
  ~TestingResults(){}
  TestingResults(const TestingResults& r)
  {
    this->success = r.success;    
    this->failure = r.failure;
    this->skip    = r.skip;
    this->expected_failure   = r.expected_failure;
    this->unexpected_success = r.unexpected_success;
  }

  void init()
  {
    this->success = 0;    
    this->failure = 0;
    this->skip    = 0;
    this->expected_failure   = 0;
    this->unexpected_success = 0;
  }

  TestingResults& operator+=(const TestingResults& rhs)
  {
    this->success += rhs.success;    
    this->failure += rhs.failure;
    this->skip    += rhs.skip;
    this->expected_failure   += rhs.expected_failure;
    this->unexpected_success += rhs.unexpected_success;
    
    return *this;
  }

  TestingResults& operator+=(const TestOutcome outcome)
  {
    switch(outcome)
    {
      case PASS:
        this->success++;
        break;
      case FAIL:
        this->failure++;
        break;
      case SKIP:
        this->skip++;
        break;
      case EXPECTED_FAIL:
        this->expected_failure++;
        break;
      case UNEXPECTED_PASS:
        this->unexpected_success++;
        break;
      default:
        std::cout << "Warning: Unrecognized test outcome code " << outcome << ". Assuming failure ...\n";
        this->failure++;
    }
    return *this;
  }

  int summary()
  {
    std::cout << "\nTest Summary\n";
    // std::cout << "----------------------------\n";
    std::cout << "\tSuccessful tests:     " << success            << "\n"; 
    std::cout << "\tFailed test:          " << failure            << "\n";
    std::cout << "\tSkipped tests:        " << skip               << "\n";
    std::cout << "\tExpected failures:    " << expected_failure   << "\n";
    std::cout << "\tUnexpected successes: " << unexpected_success << "\n";
    std::cout << "\n";

    return failure;
  }
};

TestingResults operator+(const TestingResults& lhs, const TestingResults& rhs)
{
  return TestingResults(lhs) += rhs;
}

TestingResults operator+(const TestingResults& lhs, const TestOutcome outcome)
{
  return TestingResults(lhs) += outcome;
}

TestingResults operator+(const TestOutcome outcome, const TestingResults& rhs)
{
  return TestingResults(rhs) += outcome;
}


static const real_type zero = 0.0;
static const real_type quarter = 0.25;
static const real_type half = 0.5;
static const real_type one = 1.0;
static const real_type two = 2.0;
static const real_type three = 3.0;

/// @brief eps = 2.2e-15 for double type
static const real_type eps = 10*std::numeric_limits<real_type>::epsilon();


class TestBase
{
public:
  TestBase() = default;

protected:
  /// Returns true if two real numbers are equal within tolerance
  //[[nodiscard]] 
  static bool isEqual(const real_type a, const real_type b)
  {
    return (std::abs(a - b)/(1.0 + std::abs(b)) < eps);
  }
};

}} // namespace ReSolve::tests
