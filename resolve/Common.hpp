#pragma once

#include <limits>

//TODO: temporary
#include <cstdint>

namespace ReSolve {


  // TODO: these should be dropped
  constexpr double EPSILON = 1.0e-18;
  constexpr double EPSMAC  = 1.0e-16;


  // TODO: let cmake manage these. combined with the todo above relating to cstdint
  //       this is related to resolve/lusol/lusol_precision.f90. whatever is here should
  //       have an equivalent there

  // NOTE: i'd love to make this std::float64_t but we're not on c++23
  using real_type = double;
  using index_type = std::int32_t;

  namespace constants
  {
    constexpr real_type ZERO = 0.0;
    constexpr real_type ONE = 1.0;
    constexpr real_type MINUSONE = -1.0;
    constexpr real_type DEFAULT_TOL = 100*std::numeric_limits<real_type>::epsilon();
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
