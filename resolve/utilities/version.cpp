#include <unordered_map>
#include <string>

#include "version.hpp"
#include <resolve/resolve_defs.hpp>

// Function that splits the verison in major minor and patch ints
int ReSolveVersionGetVersion(int *major, int *minor, int *patch) {
  *major = atoi(RESOLVE_VERSION_MAJOR);
  *minor = atoi(RESOLVE_VERSION_MINOR);
  *patch = atoi(RESOLVE_VERSION_PATCH);
  return 0;
}

// Function that grabs ReSolves Version as a string
int ReSolveVersionGetVersionStr(std::string &str) {
  str = RESOLVE_VERSION;
  return 0;
}

