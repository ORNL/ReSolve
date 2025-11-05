#include <iostream>
#include <string>

#include <resolve/Common.hpp>
#include <resolve/utilities/version/version.hpp>

// author: RD
// version test to check to make sure ReSolve's version can be printed

/**
 * @brief Test ReSolve version
 *
 * The purpose of this mildly annoying test is to force developers
 * to change version at two different places. The hope is this test
 * will fail if the version is changed accidentally.
 *
 * @return int If test was successful return zero
 */
int main()
{
  using namespace ReSolve::colors;
  std::string answer("0.99.2");
  std::string versionstr;
  ReSolve::VersionGetVersionStr(versionstr);
  std::cout << "ReSolveVersionGetVersionStr Test: " << versionstr << std::endl
            << std::endl;

  if (versionstr != answer)
  {
    std::cout << "ReSolve version set incorrectly. "
              << "Test " << RED << "FAILED" << CLEAR << "\n";
    return 1;
  }
  std::cout << "ReSolve version test " << GREEN << "PASSED" << CLEAR << "\n";

  return 0;
}
