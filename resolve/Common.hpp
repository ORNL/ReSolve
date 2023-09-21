#pragma once

namespace ReSolve {


  constexpr double EPSILON = 1.0e-18;
  constexpr double EPSMAC  = 1.0e-16;
  
    
  using real_type = double;
  using index_type = int;
  
  namespace constants
  {
    constexpr real_type ZERO = 0.0;
    constexpr real_type ONE = 1.0;
    constexpr real_type MINUSONE = -1.0;
  }

} // namespace ReSolve
