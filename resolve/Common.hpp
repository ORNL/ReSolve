#pragma once

#include <limits>

#include <resolve/resolve_defs.hpp>

namespace ReSolve
{

  namespace constants
  {
    constexpr real_type  ZERO      = 0.0;
    constexpr real_type  ONE       = 1.0;
    constexpr real_type  TWO       = 2.0;
    constexpr real_type  HALF      = 0.5;
    constexpr real_type  MINUS_ONE = -1.0;
    constexpr index_type SEED      = 12345;

    constexpr real_type MACHINE_EPSILON = std::numeric_limits<real_type>::epsilon();
  } // namespace constants

  namespace colors
  {
    // must be const pointer and const dest for
    // const string declarations to pass -Wwrite-strings
    static const char* const RED    = "\033[1;31m";
    static const char* const GREEN  = "\033[1;32m";
    static const char* const YELLOW = "\033[33;1m";
    static const char* const BLUE   = "\033[34;1m";
    static const char* const ORANGE = "\u001b[38;5;208m";
    static const char* const CLEAR  = "\033[0m";
  } // namespace colors

} // namespace ReSolve
