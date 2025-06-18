#include "version.hpp"

#include <string>
#include <unordered_map>

#include <resolve/resolve_defs.hpp>

namespace ReSolve
{
  // Function that splits the verison in major minor and patch ints
  int VersionGetVersion(int* major, int* minor, int* patch)
  {
    *major = atoi(RESOLVE_VERSION_MAJOR);
    *minor = atoi(RESOLVE_VERSION_MINOR);
    *patch = atoi(RESOLVE_VERSION_PATCH);
    return 0;
  }

  // Function that grabs ReSolves Version as a string
  int VersionGetVersionStr(std::string& str)
  {
    str = RESOLVE_VERSION;
    return 0;
  }

} // namespace ReSolve
