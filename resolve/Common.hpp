#pragma once

#include <limits>

//TODO: temporary
#include <cstdint>

namespace ReSolve {

  /// @todo Provide CMake option to se these types at config time
  using real_type = double;
  using index_type = std::int32_t;

  namespace constants
  {
    constexpr real_type ZERO = 0.0;
    constexpr real_type ONE = 1.0;
    constexpr real_type MINUSONE = -1.0;
    constexpr real_type MACHINE_EPSILON  = std::numeric_limits<real_type>::epsilon();
    constexpr real_type DEFAULT_TOL = 100 * MACHINE_EPSILON;
    // constexpr real_type LOOSE_TOL = 1000 * DEFAULT_TOL;
    // constexpr real_type REDO_FACTOR_TOL = 1e-7;
    // constexpr real_type DEFAULT_ZERO_DIAGONAL = 1e-6;
  }

  namespace colors
  {
    // must be const pointer and const dest for
    // const string declarations to pass -Wwrite-strings
    static const char * const  RED       = "\033[1;31m";
    static const char * const  GREEN     = "\033[1;32m";
    static const char * const  YELLOW    = "\033[33;1m";
    static const char * const  BLUE      = "\033[34;1m";
    static const char * const  ORANGE    = "\u001b[38;5;208m";
    static const char * const  CLEAR     = "\033[0m";
  }

} // namespace ReSolve
