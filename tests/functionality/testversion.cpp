#include <string>
#include <iostream>


#include <resolve/utilities/version.hpp>
#include <resolve/utilities/version.cpp>

//author: RD
//version test to check to make sure ReSolve's version can be printed


int main(int argc, char *argv[])
{
  std::string versionstr;
  ReSolveVersionGetVersionStr(versionstr);
  std::cout<<"ReSolveVersionGetVersionStr Test: "<<versionstr<<std::endl<<std::endl;;
  
  return 0;
}