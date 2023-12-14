#include <string>
#include <iostream>


#include <resolve/utilities/version/version.hpp>

//author: RD
//version test to check to make sure ReSolve's version can be printed

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
  std::string answer("0.99.1");
  std::string versionstr;
  ReSolve::VersionGetVersionStr(versionstr);
  std::cout << "ReSolveVersionGetVersionStr Test: " << versionstr << std::endl << std::endl;

  if (versionstr != answer) {
    std::cout << "ReSolve version set incorrectly!\n";
    return 1;
  }
  
  return 0;
}