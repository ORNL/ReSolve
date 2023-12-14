#pragma once

namespace ReSolve
{

  /**
   * Sets major, minor, and patch versions for current ReSolve build. The user is
   * responsible for free'ing this memory.
   */
  int VersionGetVersion(int *, int *, int *);

  /**
   * Sets string with build version for current ExaGO build in format
   * "major.minor.patch". The user is responsible for free'ing this memory.
   */
  int VersionGetVersionStr(std::string &);

} 
